#!/usr/bin/env python3
"""Verify a DFlash --derive copy with stdlib only; no model/backend imports."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

CHUNK = 1024 * 1024
HEADER_LIMIT = 256 * 1024 * 1024
# (elements per block, bytes per block), checked against ggml.h/ggml-common.h
# at llama.cpp de5122d879d00d2895c36be7e0cee94105aeeabd. Other types fail closed.
DTYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 30: (1, 2)}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    s = path.stat()
    return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def sha_text(value):
    need(re.fullmatch(r'[0-9a-f]{64}', value) is not None, 'expected lowercase SHA-256')
    return value


def digest_range(path, offset, size, expected_identity):
    need(type(offset) is int and type(size) is int and offset >= 0 and size > 0,
         'invalid byte range')
    need(offset + size <= expected_identity['size'], 'range exceeds file')
    need(identity(path) == expected_identity, 'file changed before read: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        st = os.fstat(f.fileno())
        need((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == tuple(
            expected_identity[k] for k in ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')),
            'file changed at open')
        f.seek(offset)
        left = size
        while left:
            part = f.read(min(left, CHUNK))
            need(part, 'short read')
            h.update(part)
            left -= len(part)
        after = os.fstat(f.fileno())
        need(
            (after.st_dev, after.st_ino, after.st_size,
             after.st_mtime_ns, after.st_ctime_ns)
            == (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns),
            'file changed during read',
        )
    need(identity(path) == expected_identity, 'file changed after read: ' + str(path))
    return h.hexdigest()


class HeaderReader:
    def __init__(self, stream, size):
        self.f, self.limit = stream, min(size, HEADER_LIMIT)

    def read(self, size):
        need(0 <= size <= self.limit - self.f.tell(), 'GGUF header exceeds bound')
        b = self.f.read(size)
        need(len(b) == size, 'truncated GGUF header')
        return b

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, decode=True):
        size = self.number('Q')
        need(size <= 16 * 1024 * 1024, 'GGUF string exceeds bound')
        b = self.read(size)
        return b.decode('utf-8') if decode else None

    def value(self, kind, array=False):
        if kind in SCALARS:
            return self.number(SCALARS[kind])
        if kind == 8:
            return self.string()
        need(kind == 9 and not array, 'unsupported GGUF value/nested array')
        element, count = self.number('I'), self.number('Q')
        need(count <= 8_000_000, 'GGUF array exceeds bound')
        if element in SCALARS:
            total = count * struct.calcsize(SCALARS[element])
            need(total <= self.limit - self.f.tell(), 'GGUF array exceeds header')
            self.f.seek(total, 1)
        elif element == 8:
            for _ in range(count):
                self.string(decode=False)
        else:
            raise ValueError('unsupported GGUF array element type')
        return None


def header(path, file_id):
    metadata, scalar, tensors = {}, {}, []
    with path.open('rb') as f:
        r = HeaderReader(f, file_id['size'])
        need(r.read(4) == b'GGUF' and r.number('I') == 3, 'only GGUF v3 is supported')
        count, kv_count = r.number('Q'), r.number('Q')
        need(0 < count <= 100_000 and kv_count <= 100_000, 'GGUF count exceeds bound')
        for _ in range(kv_count):
            key = r.string()
            need(
                key and '\x00' not in key and key not in metadata,
                'duplicate/invalid metadata key',
            )
            start = f.tell()
            kind = r.number('I')
            value = r.value(kind)
            end = f.tell()
            metadata[key] = {'type': kind, 'bytes': end - start,
                             'sha256': digest_range(path, start, end - start, file_id)}
            scalar[key] = value
        alignment = scalar.get('general.alignment', 32)
        need(
            type(alignment) is int
            and 1 <= alignment <= 65536
            and alignment & (alignment - 1) == 0,
            'unsupported GGUF alignment',
        )
        names = set()
        for _ in range(count):
            name, rank = r.string(), r.number('I')
            need(name and '\x00' not in name and name not in names and 1 <= rank <= 4,
                 'duplicate/invalid tensor name or rank')
            names.add(name)
            shape = [r.number('Q') for _ in range(rank)]
            kind, offset = r.number('I'), r.number('Q')
            need(kind in DTYPES, f'unsupported tensor dtype {kind}: {name}')
            block, block_bytes = DTYPES[kind]
            elements = math.prod(shape)
            need(all(0 < x <= (1 << 32) for x in shape) and elements <= (1 << 63)
                 and shape[0] % block == 0, 'invalid tensor dimensions/block width')
            size = elements // block * block_bytes
            need(offset % alignment == 0, 'unaligned tensor offset')
            tensors.append(dict(
                name=name,
                shape=shape,
                dtype=kind,
                relative_offset=offset,
                bytes=size,
            ))
        data = (f.tell() + alignment - 1) // alignment * alignment
        need(data <= file_id['size'], 'tensor data starts outside file')
    for t in tensors:
        t['offset'] = data + t['relative_offset']
        need(t['offset'] + t['bytes'] <= file_id['size'], 'tensor exceeds file: ' + t['name'])
    ordered = sorted(tensors, key=lambda t: t['offset'])
    need(all(a['offset'] + a['bytes'] <= b['offset'] for a, b in zip(ordered, ordered[1:])),
         'overlapping tensors')
    need(identity(path) == file_id, 'file changed during header read')
    return dict(
        metadata=metadata,
        scalar=scalar,
        tensors=tensors,
        data_offset=data,
        alignment=alignment,
    )


def verify(args):
    paths = {k: getattr(args, k).resolve(strict=True) for k in ('target', 'draft', 'derived')}
    ids = {k: identity(p) for k, p in paths.items()}
    need(
        len({(v['device'], v['inode']) for v in ids.values()}) == 3,
        'source and derived files must be distinct',
    )
    need(ids['derived']['size'] <= 10 * 1024**3, 'derived file exceeds producer 10 GiB limit')
    manifest_path = args.manifest.resolve(strict=True)
    manifest_id = identity(manifest_path)
    need(0 < manifest_id['size'] <= 16 * 1024 * 1024, 'manifest exceeds bound')
    manifest_hash = digest_range(manifest_path, 0, manifest_id['size'], manifest_id)
    if args.manifest_sha256:
        need(manifest_hash == sha_text(args.manifest_sha256), 'manifest SHA-256 mismatch')
    manifest = json.loads(manifest_path.read_text())
    need(manifest.get('raw_quantized_bytes_unchanged') is True, 'manifest copy claim missing')
    location = manifest.get('derived')
    location = location.get('path') if isinstance(location, dict) else location
    need(
        isinstance(location, str) and Path(location).resolve() == paths['derived'],
        'manifest derived path differs',
    )
    headers = {k: header(p, ids[k]) for k, p in paths.items()}
    draft, derived, target = (headers[k] for k in ('draft', 'derived', 'target'))
    need(draft['scalar'].get('general.architecture') == 'dflash', 'expected DFlash draft')
    need(draft['metadata'] == derived['metadata'], 'derived metadata differs from draft')
    need(not any(k.startswith('split.') for h in headers.values() for k in h['metadata']),
         'split sources are not supported by this helper')
    shared = ['token_embd.weight', 'output.weight']
    dt = {t['name']: t for t in draft['tensors']}
    tt = {t['name']: t for t in target['tensors']}
    need(not set(shared) & set(dt) and set(shared) <= set(tt), 'missing/duplicate shared tensors')
    width = draft['scalar'].get('dflash.embedding_length')
    need(
        type(width) is int
        and width > 0
        and all(
            len(tt[n]['shape']) == 2 and tt[n]['shape'][0] == width
            for n in shared
        )
        and tt[shared[0]]['shape'][1] == tt[shared[1]]['shape'][1],
        'shared width/vocabulary differs',
    )
    expected = [('draft', t) for t in draft['tensors']] + [('target', tt[n]) for n in shared]
    need([t['name'] for t in derived['tensors']] == [t['name'] for _, t in expected],
         'derived tensor set/order differs; '
         'require every draft tensor and exactly two shared tensors')
    copies = manifest.get('tensor_copies')
    need(
        isinstance(copies, list) and len(copies) == len(expected),
        'manifest omits/adds tensor ranges',
    )
    hashes = {}
    for key, path in paths.items():
        expected_hash = sha_text(getattr(args, key + '_sha256'))
        hashes[key] = digest_range(path, 0, ids[key]['size'], ids[key])
        need(hashes[key] == expected_hash, key + ' full SHA-256 mismatch')
    checked = []
    for (source, original), out, item in zip(expected, derived['tensors'], copies):
        need(
            isinstance(item, dict) and item.get('name') == original['name'] == out['name'],
            'manifest tensor name/order differs',
        )
        need(
            isinstance(item.get('source'), str)
            and Path(item['source']).resolve() == paths[source],
            'manifest source differs',
        )
        need(
            (original['shape'], original['dtype'], original['bytes'])
            == (out['shape'], out['dtype'], out['bytes']),
            'derived tensor shape/dtype differs',
        )
        for key, value in [
            ('source_offset', original['offset']),
            ('derived_offset', out['offset']),
            ('bytes', original['bytes']),
        ]:
            need(
                type(item.get(key)) is int and item[key] == value,
                'manifest range differs: ' + key,
            )
        a = digest_range(paths[source], original['offset'], original['bytes'], ids[source])
        b = digest_range(paths['derived'], out['offset'], out['bytes'], ids['derived'])
        need(a == b, 'tensor bytes differ: ' + original['name'])
        if 'raw_sha256' in item:
            need(sha_text(item['raw_sha256']) == a, 'manifest tensor SHA-256 differs')
        checked.append(dict(name=original['name'], source=source, source_offset=original['offset'],
                            derived_offset=out['offset'], bytes=original['bytes'], sha256=a,
                            shape=original['shape'], dtype=original['dtype']))
    need(
        all(identity(p) == ids[k] for k, p in paths.items())
        and identity(manifest_path) == manifest_id,
        'input identity changed during verification',
    )
    return dict(
        passed=True,
        files={
            k: dict(path=str(p), identity=ids[k], sha256=hashes[k])
            for k, p in paths.items()
        },
        manifest=dict(
            path=str(manifest_path), identity=manifest_id, sha256=manifest_hash,
        ),
        tensor_count=len(checked),
        copied_tensor_bytes=sum(t['bytes'] for t in checked),
        tensor_copies=checked,
        metadata_equal=True,
        model_inference_run=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('target', 'draft', 'derived'):
        parser.add_argument('--' + role, type=Path, required=True)
        parser.add_argument('--' + role + '-sha256', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    need(not args.receipt.exists(), 'refuse to overwrite receipt')
    result = dict(passed=False, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        result.update(verify(args))
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    with args.receipt.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps(dict(
        passed=result['passed'],
        receipt=str(args.receipt),
        error=result.get('error'),
    )))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
