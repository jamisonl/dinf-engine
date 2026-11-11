#!/usr/bin/env python3
"""Build the pinned CPU-reference repair without changing an installed runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
POLICY = 'ggml-cpu-flash-f32-v2'
REVISION = '15fa3c493bfcd040b5f4dcb29e1c998a0846de16'
ACTIVATIONS_POLICY = 'ggml-cpu-f32-activations-v3'
QUERY_POLICY = 'ggml-cpu-flash-f32-query-v4'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def file_record(path):
    path = path.resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def supported_host(system, machine, version, brand):
    require(system == 'Darwin' and machine == 'arm64', 'Only macOS arm64 is supported')
    require(version and int(version.split('.')[0]) >= 26, 'macOS26 or later required')
    require(brand.startswith(('Apple M4', 'Apple M5')), 'Only tested Apple M4/M5 CPU families are supported')


def config_check(text, expected):
    for name, value in expected.items():
        found = re.search(r'^set\(' + re.escape(name) + r'\s+"([^"\n]*)"\)', text, re.M)
        require(found is not None and found.group(1) == value, 'Unsupported installed config: ' + name)


def pin_check(pins):
    for name, value in pins.items():
        require(sha(name) == value, 'Pinned file changed: ' + name)


def new_output_path(output, source, prefix):
    # Resolve existing parents too: /tmp and symlink aliases must not bypass
    # the source/runtime containment check when the output leaf is new.
    output = output.resolve()
    require(not output.exists(), 'Output must be new')
    require(not output.is_relative_to(source.resolve())
            and not output.is_relative_to(prefix.resolve()),
            'Output must not be inside source/runtime')
    return output


def activation_policy(enabled):
    if not enabled:
        return None
    notice = json.loads((HERE / 'F32_ACTIVATIONS.json').read_text())
    require(notice['policy'] == ACTIVATIONS_POLICY and notice['extends'] == POLICY
            and notice['source_revision'] == REVISION, 'Unexpected activation policy')
    require(len(notice['patches']) == 2, 'Expected exactly the softplus and F32 GELU repairs')
    for patch in notice['patches']:
        require(Path(patch['file']).name == patch['file'], 'Patch must be a local basename')
        require(sha(HERE / patch['file']) == patch['sha256'], 'Activation patch hash differs')
    return notice


def query_policy(enabled):
    if not enabled:
        return None
    notice = json.loads((HERE / 'F32_QUERY.json').read_text())
    require(notice['policy'] == QUERY_POLICY and notice['extends'] == POLICY
            and notice['source_revision'] == REVISION, 'Unexpected F32 query policy')
    require(len(notice['patches']) == 1, 'Expected exactly one F32 query repair')
    repair = notice['patches'][0]
    require(repair['file'] == 'ggml-cpu-flash-f32-query-v4.patch'
            and repair['source_file'] == 'src/ggml-cpu/ops.cpp', 'Unexpected F32 query source')
    require(sha(HERE / repair['file']) == repair['sha256'], 'F32 query patch hash differs')
    return notice


def select_policy(activations, query):
    require(not (activations and query), 'Choose one optional CPU reference profile')
    return QUERY_POLICY if query else ACTIVATIONS_POLICY if activations else POLICY


def prepare(args):
    notice = json.loads((HERE / 'SOURCE.json').read_text())
    require(notice['policy'] == POLICY and notice['source_revision'] == REVISION, 'Unexpected policy/revision')
    patch = HERE / 'ggml-cpu-flash-f32-v2.patch'
    require(sha(patch) == notice['source_patch_sha256'], 'Patch hash differs')
    require(sha(HERE / 'LICENSE.ggml') == notice['license']['sha256'], 'License hash differs')
    brand = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    supported_host(platform.system(), platform.machine(), platform.mac_ver()[0], brand)
    source = args.ggml_source.resolve(strict=True)
    prefix = args.ggml_prefix.resolve(strict=True)
    omp = args.openmp_prefix.resolve(strict=True)
    cmake = Path(shutil.which(args.cmake) or args.cmake).resolve(strict=True)
    compiler = Path('/usr/bin/clang++')
    compiler_version = subprocess.check_output([str(compiler), '--version'], text=True)
    require(compiler_version.startswith('Apple clang version 21.'), 'Tested Apple clang21 required')
    config = prefix / 'lib/cmake/ggml/ggml-config.cmake'
    config_check(config.read_text(), notice['required_config'])
    pins = {str(p.resolve()): sha(p) for p in HERE.iterdir() if p.is_file()}
    for rel, expected in notice['source_files_sha256'].items():
        file = source / rel
        require(file.resolve(strict=True).is_relative_to(source), 'Source path escaped root')
        require(file.is_file() and not file.is_symlink(), 'Regular source file required: ' + rel)
        require(sha(file) == expected, 'Source hash differs: ' + rel)
        pins[str(file.resolve())] = expected
    runtime = {}
    for rel, expected in notice['runtime_reference'].items():
        file = prefix / rel
        require(sha(file) == expected, 'Installed runtime differs from tested pinned build: ' + rel)
        runtime[str(file.resolve())] = expected
    for p in (prefix / 'lib/cmake/ggml').glob('*.cmake'):
        pins[str(p.resolve())] = sha(p)
    for p in [cmake, compiler, Path('/usr/bin/clang'), omp / 'lib/libomp.dylib']:
        pins[str(p.resolve())] = sha(p)
    runtime[str((omp / 'lib/libomp.dylib').resolve())] = sha(omp / 'lib/libomp.dylib')
    output = new_output_path(args.output, source, prefix)
    return notice, source, prefix, omp, cmake, pins, runtime, output, brand, compiler_version


def run_logged(command, log, env):
    with log.open('xb') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Command failed; inspect ' + str(log))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ggml-source', type=Path, required=True, help='ggml subdirectory of the exact pinned llama.cpp source')
    ap.add_argument('--ggml-prefix', type=Path, required=True, help='Existing pinned ggml0.10.0 installation')
    ap.add_argument('--openmp-prefix', type=Path, required=True)
    ap.add_argument('--cmake', default='cmake')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--check', action='store_true', help='Verify inputs only; do not create/build output')
    ap.add_argument('--f32-activations', action='store_true', help='Also repair CPU softplus and F32 GELU; distinct v3 policy, not the six-model v2 reference')
    ap.add_argument('--original-f32-query', action='store_true', help='Keep original F32 query values in requested-F32 flash attention; separate v4 profile, not v2/v3')
    args = ap.parse_args()
    selected_policy = select_policy(args.f32_activations, args.original_f32_query)
    notice, source, prefix, omp, cmake, pins, runtime, output, brand, compiler_version = prepare(args)
    activations = activation_policy(args.f32_activations)
    query = query_policy(args.original_f32_query)
    pin_check(pins | runtime)
    if args.check:
        print(json.dumps({'inputs_verified': True, 'source_files': len(notice['source_files_sha256']), 'policy': selected_policy, 'built': False}))
        return
    output.mkdir(parents=True)
    before = {'policy': selected_policy, 'source_revision': REVISION, 'source_patch_sha256': notice['source_patch_sha256'],
              'source_pins': pins, 'runtime_pins': runtime, 'platform': 'darwin-arm64', 'isa': 'apple_m4',
              'cpu_brand': brand, 'compiler_version': compiler_version, 'models_loaded': 0, 'gpu_calls': 0}
    if activations:
        before['activation_policy'] = activations
    if query:
        before['query_policy'] = query
    write_new(output / 'before.json', before)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DINF_', 'CANDLE_', 'GGML_', 'DYLD_', 'LD_'))
           and k not in ['CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS']}
    try:
        copied = output / 'source'
        for rel in notice['source_files_sha256']:
            dst = copied / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dst)
        shutil.copyfile(HERE / 'CMakeLists.txt', copied / 'src/CMakeLists.txt')
        patch_cmd = ['/usr/bin/patch', '-t', '-N', '-p2', '-d', str(copied), '-i', str(HERE / 'ggml-cpu-flash-f32-v2.patch')]
        run_logged(patch_cmd, output / 'patch.log', env)
        require(sha(copied / 'src/ggml-cpu/ops.cpp') == notice['patched_ops_sha256'], 'Patched source hash differs')
        activation_commands = []
        if activations:
            for repair in activations['patches']:
                source_file = copied / repair['source_file']
                require(source_file.resolve().is_relative_to(copied.resolve()), 'Activation source escaped root')
                require(sha(source_file) == repair['original_sha256'], 'Activation source hash differs')
                command = ['/usr/bin/patch', '-t', '-N', '-p2', '-d', str(copied), '-i', str(HERE / repair['file'])]
                run_logged(command, output / (repair['file'] + '.log'), env)
                require(sha(source_file) == repair['patched_sha256'], 'Patched activation source hash differs')
                activation_commands.append(command)

        query_commands = []
        if query:
            for repair in query['patches']:
                source_file = copied / repair['source_file']
                require(sha(source_file) == repair['original_sha256'], 'F32 query source hash differs')
                command = ['/usr/bin/patch', '-t', '-N', '-p2', '-d', str(copied), '-i', str(HERE / repair['file'])]
                run_logged(command, output / (repair['file'] + '.log'), env)
                require(sha(source_file) == repair['patched_sha256'], 'Patched F32 query source hash differs')
                query_commands.append(command)

        build = output / 'build'
        configure = [str(cmake), '-S', str(copied / 'src'), '-B', str(build),
                     '-DDINF_GGML_PREFIX=' + str(prefix), '-DOpenMP_ROOT=' + str(omp),
                     '-DCMAKE_C_COMPILER=/usr/bin/clang', '-DCMAKE_CXX_COMPILER=/usr/bin/clang++']
        compile_cmd = [str(cmake), '--build', str(build), '--target', 'ggml-cpu-apple_m4', '--parallel', '1']
        run_logged(configure, output / 'configure.log', env)
        run_logged(compile_cmd, output / 'build.log', env)
        plugin = build / 'plugins/libggml-cpu-apple_m4.so'
        linked = subprocess.check_output(['/usr/bin/otool', '-L', str(plugin)], text=True)
        libs = [line.strip().split(' (')[0] for line in linked.splitlines()[1:]]
        base = (prefix / 'lib/libggml-base.0.10.0.dylib').resolve()
        base_links = [Path(p) for p in libs if 'libggml-base' in Path(p).name]
        require(len(base_links) == 1 and base_links[0].resolve() == base, 'Plugin must link exactly the unchanged installed base')
        require(not any('metal' in Path(p).name.lower() for p in libs), 'CPU plugin must not link Metal')
        commands = (build / 'compile_commands.json').read_text()
        require('-march=armv9.2-a+dotprod+i8mm+nosve+sme' in commands, 'Unexpected ISA')
        require('-ffast-math' not in commands and '-Ofast' not in commands, 'Unsupported floating-point flags')
        pin_check(pins | runtime)
        copied_pins = {str(p.resolve()): sha(p) for p in copied.rglob('*') if p.is_file()}
        receipts = {str(p.resolve()): sha(p) for p in [output/'patch.log', output/'configure.log', output/'build.log', build/'compile_commands.json', build/'CMakeCache.txt']}
        result = before | {'passed': True, 'qualifying': False, 'build_parallelism': 1,
                           'source_pins': pins | copied_pins, 'build_pins': receipts,
                           'blas_plugin': file_record(prefix / 'libexec/libggml-blas.so'),
                           'stock_plugin': file_record(prefix / 'libexec/libggml-cpu-apple_m4.so'),
                           'base_library': file_record(base), 'linked_libraries': libs,
                           'builds': {'candidate': {'plugin': file_record(plugin)}},
                           'commands': [patch_cmd, *activation_commands, *query_commands, configure, compile_cmd],
                           'scope': 'Built CPU reference plugin only. Requires separate fixed-input/model qualification; no engine or numeric-gate change.'}
        write_new(output / 'build-receipt.json', result)
        print(json.dumps({'passed': True, 'receipt': str(output/'build-receipt.json'), 'plugin': result['builds']['candidate']['plugin']}))
    except BaseException as error:
        write_new(output / 'failure.json', {'passed': False, 'error': repr(error), 'policy': selected_policy})
        raise

if __name__ == '__main__':
    main()
