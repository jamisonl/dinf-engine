from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest

P = Path(__file__).with_name('verify_dflash_copy.py')
spec = importlib.util.spec_from_file_location('verifier', P)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def txt(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b


def write_gguf(path, arch, tensors, width=4):
    kv = [
        ('general.architecture', 8, txt(arch)),
        ('dflash.embedding_length', 4, struct.pack('<I', width)),
        ('dflash.target_layers', 9, struct.pack('<IQII', 4, 2, 1, 2)),
    ]
    header = (
        b'GGUF'
        + struct.pack('<IQQ', 3, len(tensors), len(kv))
        + b''.join(
            txt(k) + struct.pack('<I', kind) + value
            for k, kind, value in kv
        )
    )
    offset = 0
    offset_fields = []
    rows = []
    for name, shape, kind, raw in tensors:
        header += (
            txt(name)
            + struct.pack('<I', len(shape))
            + b''.join(struct.pack('<Q', x) for x in shape)
            + struct.pack('<I', kind)
        )
        offset_fields.append(len(header))
        header += struct.pack('<Q', offset)
        rows.append((name, shape, kind, offset, raw))
        offset = (offset + len(raw) + 31) // 32 * 32
    start = (len(header) + 31) // 32 * 32
    data = bytearray(header + b'\x00' * (start - len(header)))
    for _, _, _, off, raw in rows:
        if len(data) < start + off:
            data.extend(b'\x00' * (start + off - len(data)))
        data.extend(raw)
    path.write_bytes(data)
    return {'data': start, 'rows': rows, 'offset_fields': offset_fields}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Verify(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = Path(self.tmp.name)
        self.draft = [
            ('fc.weight', [4, 4], 0, struct.pack('<16f', *range(16))),
            ('enc.output_norm.weight', [4], 0, struct.pack('<4f', 1, 2, 3, 4)),
        ]
        self.shared = [
            ('token_embd.weight', [4, 3], 0, struct.pack('<12f', *range(12))),
            ('output.weight', [4, 3], 0, struct.pack('<12f', *range(12, 24))),
        ]
        self.args = argparse.Namespace(
            target=self.b / 'target.gguf',
            draft=self.b / 'draft.gguf',
            derived=self.b / 'derived.gguf',
            manifest=self.b / 'copy.json',
            manifest_sha256=None,
        )
        t = write_gguf(self.args.target, 'qwen35', self.shared)
        d = write_gguf(self.args.draft, 'dflash', self.draft)
        o = write_gguf(self.args.derived, 'dflash', self.draft + self.shared)
        copies = []
        for i, (name, shape, kind, off, raw) in enumerate(o['rows']):
            source = self.args.draft if i < 2 else self.args.target
            layout = d if i < 2 else t
            row = layout['rows'][i if i < 2 else i - 2]
            copies.append(dict(
                name=name,
                source=str(source),
                source_offset=layout['data'] + row[3],
                derived_offset=o['data'] + off,
                bytes=len(raw),
                raw_sha256=hashlib.sha256(raw).hexdigest(),
            ))
        self.manifest = dict(
            derived=str(self.args.derived),
            raw_quantized_bytes_unchanged=True,
            tensor_copies=copies,
        )
        self.args.manifest.write_text(json.dumps(self.manifest))
        self.refresh()
        self.derived_layout = o

    def tearDown(self):
        self.tmp.cleanup()

    def refresh(self):
        for name in ['target', 'draft', 'derived']:
            setattr(self.args, name + '_sha256', sha(getattr(self.args, name)))

    def write_manifest(self):
        self.args.manifest.write_text(json.dumps(self.manifest))

    def test_valid_complete_copy(self):
        r = v.verify(self.args)
        self.assertTrue(r['passed'])
        self.assertEqual(r['tensor_count'], 4)
        self.assertEqual(r['copied_tensor_bytes'], 176)

    def test_missing_manifest_range(self):
        self.manifest['tensor_copies'].pop()
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'omits/adds'):
            v.verify(self.args)

    def test_duplicate_manifest_name(self):
        self.manifest['tensor_copies'][1] = self.manifest['tensor_copies'][0]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'name/order'):
            v.verify(self.args)

    def test_wrong_source(self):
        self.manifest['tensor_copies'][0]['source'] = str(self.args.target)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'source differs'):
            v.verify(self.args)

    def test_false_range_offset(self):
        self.manifest['tensor_copies'][0]['source_offset'] += 4
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'range differs'):
            v.verify(self.args)

    def test_bad_full_hash(self):
        self.args.target_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'full SHA-256'):
            v.verify(self.args)

    def test_bad_manifest_hash(self):
        self.args.manifest_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'manifest SHA-256'):
            v.verify(self.args)

    def test_changed_metadata(self):
        write_gguf(self.args.derived, 'dflash', self.draft + self.shared, width=8)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            v.verify(self.args)

    def test_changed_derived_tensor_bytes_even_with_new_full_hash(self):
        data = bytearray(self.args.derived.read_bytes())
        data[self.derived_layout['data']] ^= 1
        self.args.derived.write_bytes(data)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'tensor bytes differ'):
            v.verify(self.args)

    def test_missing_derived_tensor(self):
        write_gguf(self.args.derived, 'dflash', self.draft + self.shared[:1])
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'tensor set/order'):
            v.verify(self.args)

    def test_changed_derived_dtype(self):
        tensors = copy.deepcopy(self.draft + self.shared)
        name, shape, kind, raw = tensors[0]
        tensors[0] = (name, shape, 1, raw[:32])
        write_gguf(self.args.derived, 'dflash', tensors)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            v.verify(self.args)

    def test_unsupported_dtype(self):
        tensors = copy.deepcopy(self.draft + self.shared)
        name, shape, kind, raw = tensors[0]
        tensors[0] = (name, shape, 99, raw)
        write_gguf(self.args.derived, 'dflash', tensors)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'unsupported tensor dtype'):
            v.verify(self.args)

    def test_overlapping_ranges(self):
        data = bytearray(self.args.derived.read_bytes())
        pos = self.derived_layout['offset_fields'][1]
        data[pos:pos + 8] = struct.pack('<Q', 0)
        self.args.derived.write_bytes(data)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            v.verify(self.args)

    def test_changed_file_identity(self):
        old = v.identity(self.args.target)
        self.args.target.write_bytes(self.args.target.read_bytes() + b'\x00')
        with self.assertRaisesRegex(ValueError, 'changed before read'):
            v.digest_range(self.args.target, 0, old['size'], old)

    def test_quantized_block_lengths(self):
        draft = [
            ('fc.weight', [256, 2], 12, bytes(range(144)) * 2),
            ('enc.output_norm.weight', [256], 0, bytes(1024)),
        ]
        shared = [
            ('token_embd.weight', [256, 3], 12, bytes(range(144)) * 3),
            ('output.weight', [256, 3], 14, bytes(range(210)) * 3),
        ]
        d = write_gguf(self.args.draft, 'dflash', draft, width=256)
        t = write_gguf(self.args.target, 'qwen35', shared, width=256)
        o = write_gguf(self.args.derived, 'dflash', draft + shared, width=256)
        copies = []
        for i, (name, shape, kind, off, raw) in enumerate(o['rows']):
            src = self.args.draft if i < 2 else self.args.target
            layout = d if i < 2 else t
            sr = layout['rows'][i if i < 2 else i - 2]
            copies.append(dict(
                name=name,
                source=str(src),
                source_offset=layout['data'] + sr[3],
                derived_offset=o['data'] + off,
                bytes=len(raw),
            ))
        self.manifest['tensor_copies'] = copies
        self.write_manifest()
        self.refresh()
        r = v.verify(self.args)
        self.assertEqual(r['copied_tensor_bytes'], 2374)

    def test_partial_quantization_row_rejected(self):
        tensors = copy.deepcopy(self.draft + self.shared)
        name, shape, kind, raw = tensors[0]
        tensors[0] = (name, [4, 256], 12, bytes(576))
        write_gguf(self.args.derived, 'dflash', tensors)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'block width'):
            v.verify(self.args)

    def test_truncated_header(self):
        self.args.derived.write_bytes(b'GGUF' + struct.pack('<I', 3))
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'header exceeds bound'):
            v.verify(self.args)

    def test_enriched_manifest_compatible(self):
        self.manifest['derived'] = {
            'path': str(self.args.derived),
            'sha256': self.args.derived_sha256,
        }
        self.manifest['producer'] = {'sha256': '0' * 64}
        self.write_manifest()
        self.assertTrue(v.verify(self.args)['passed'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
