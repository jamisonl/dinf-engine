"""Check paired v2/v4 synthetic query probes. No models or third-party modules."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def raw(root, name, values):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(), 'raw path')
    data = path.read_bytes()
    require(len(data) == values * 4, 'raw extent')
    floats = array.array('f')
    floats.frombytes(data)
    require(all(math.isfinite(x) for x in floats), 'finite output')
    return data, pin(path)


def verify(control, candidate):
    old = json.loads((control / 'report.json').read_text())
    new = json.loads((candidate / 'report.json').read_text())
    expected = [(n, q) for n in (1, 3, 31, 33, 257) for q in (1, 3) if q <= n]
    for report in (old, new):
        require(report['schema_version'] == 1 and report['model_calls'] == 0
                and report['gpu_calls'] == 0 and report['threads'] == 2, 'probe scope')
        require([(c['visible'], c['rows']) for c in report['cases']] == expected, 'fixed cases')
    pins = [pin(control / 'report.json'), pin(candidate / 'report.json')]
    improvements = 0
    values = 0
    for a, b in zip(old['cases'], new['cases']):
        for key in ('id', 'visible', 'rows', 'heads', 'kv_heads', 'dim', 'physical', 'values', 'q_values_changed_by_half'):
            require(a[key] == b[key], 'case identity')
        require(b['heads'] == 4 and b['kv_heads'] == 2 and b['dim'] == 128
                and b['physical'] == ((b['visible'] + 255) // 256) * 256
                and b['values'] == b['rows'] * 4 * 128 and b['q_values_changed_by_half'] > 0, 'geometry')
        for mode in ('default', 'f32'):
            x, xp = raw(control, a[mode]['file'], a['values'])
            y, yp = raw(candidate, b[mode]['file'], b['values'])
            pins.extend((xp, yp))
            if mode == 'default':
                require(x == y, 'DEFAULT output changed')
        error = b['f32']['max_abs_to_f64_original_q']
        require(math.isfinite(error) and 0 <= error < 2e-5, 'F32 query F64 bound')
        old_rms = a['f32']['rms_to_f64_original_q']
        new_rms = b['f32']['rms_to_f64_original_q']
        require(math.isfinite(old_rms) and math.isfinite(new_rms) and old_rms >= 0 and new_rms >= 0, 'finite RMS')
        improvements += old_rms > 10 * new_rms
        values += b['values']
    require(improvements == 8, 'eight nontrivial input-precision improvements required')
    return {'passed': True, 'qualifying': False, 'model_calls': 0, 'cases': len(expected),
            'values': values, 'default_bit_identical': True, 'f64_max_abs_limit': 2e-5,
            'more_than_10x_lower_rms_cases': improvements, 'pins': pins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.control, args.candidate)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
