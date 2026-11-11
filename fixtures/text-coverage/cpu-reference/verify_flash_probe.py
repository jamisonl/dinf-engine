#!/usr/bin/env python3
"""Check saved tiny CPU flash-attention outputs against their scalar F64 rows.

This is a component check only. It does not run a model or grant a matrix pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plugin-build', type=Path, required=True)
    ap.add_argument('--result-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    build = json.loads(a.plugin_build.read_text())
    require(build['passed'] is True and build['policy'] == 'ggml-cpu-flash-f32-v2', 'repair build failed')
    plugin = build['builds']['candidate']['plugin']
    require(sha(plugin['path']) == plugin['sha256'], 'plugin changed')
    pins = build['source_pins'] | build['runtime_pins']
    pins.update({str(p.resolve()): sha(p) for p in [a.plugin_build, Path(__file__), Path(__file__).with_name('flash_probe.cpp')]})
    for path, expected in pins.items():
        require(sha(path) == expected, 'build input changed: ' + path)
    report_path = a.result_dir / 'report.json'
    report = json.loads(report_path.read_text())
    require(report['backend'] == 'CPU' and report['threads'] == 8 and report['models_loaded'] == report['gpu_backends_initialized'] == 0, 'probe scope differs')
    require(Path(report['cpu_plugin']).resolve() == Path(plugin['path']).resolve(), 'probe selected a different plugin')
    images = [Path(p) for p in report['runtime_images']]
    require([p.resolve() for p in images if p.name.startswith('libggml-cpu')] == [Path(plugin['path']).resolve()], 'CPU image count or path differs')
    require(not any(p.name == 'libggml-metal.so' for p in images), 'probe loaded Metal')
    ids = ['uniform-' + str(n) for n in [15, 16, 17, 256, 512, 1024]] + ['random-' + str(n) for n in [17, 127, 512, 1024]]
    require([c['id'] for c in report['cases']] == ids, 'fixed cases differ')
    cases = []
    for case in report['cases']:
        ident = case['id']
        root = a.result_dir / ident
        actual_bytes = (root / 'actual.f32').read_bytes()
        reference_bytes = (root / 'reference-q16.f64').read_bytes()
        require(len(actual_bytes) == 4096 * 4 and len(reference_bytes) == 4096 * 8, 'raw shape differs')
        actual = struct.unpack('<4096f', actual_bytes)
        reference = struct.unpack('<4096d', reference_bytes)
        require(all(map(math.isfinite, actual)) and all(map(math.isfinite, reference)), 'nonfinite raw output')
        error = max(abs(x-y) for x, y in zip(actual, reference))
        require(error <= 2e-6, 'fixed half-query F64 bound failed: ' + ident)
        if ident.startswith('uniform-'):
            require(all(x == 4096 for x in actual), 'constant output differs: ' + ident)
        cases.append({'id': ident, 'max_abs_to_half_query_f64': error, 'passed': True})
    raw_pins = {str(p.resolve()): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
                for p in a.result_dir.rglob('*') if p.is_file()}
    for path, expected in pins.items():
        require(sha(path) == expected, 'build input changed during review: ' + path)
    result = {'valid_evidence': True, 'component_passed': True, 'qualifying': False,
              'cases': cases, 'pins': pins, 'raw_files': raw_pins,
              'limit': 'Checks saved outputs against scalar F64 rows emitted by the pinned probe source. Does not prove a full-model gate.'}
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
