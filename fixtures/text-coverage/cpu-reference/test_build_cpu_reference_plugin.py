import importlib.util
from pathlib import Path
import unittest
import tempfile

PATH = Path(__file__).with_name('build_cpu_reference_plugin.py')
SPEC = importlib.util.spec_from_file_location('cpu_plugin_builder', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GuardTests(unittest.TestCase):
    def test_optional_profiles_stay_separate(self):
        self.assertEqual(MODULE.select_policy(False, False), MODULE.POLICY)
        self.assertEqual(MODULE.select_policy(True, False), MODULE.ACTIVATIONS_POLICY)
        self.assertEqual(MODULE.select_policy(False, True), MODULE.QUERY_POLICY)
        with self.assertRaises(ValueError):
            MODULE.select_policy(True, True)

    def test_exact_query_patch(self):
        self.assertIsNone(MODULE.query_policy(False))
        notice = MODULE.query_policy(True)
        self.assertEqual(notice['patches'][0]['source_file'], 'src/ggml-cpu/ops.cpp')

    def test_tested_host(self):
        MODULE.supported_host('Darwin', 'arm64', '27.2', 'Apple M5 Max')

    def test_linux_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.supported_host('Linux', 'aarch64', '27.2', 'Apple M5 Max')

    def test_older_isa_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.supported_host('Darwin', 'arm64', '27.2', 'Apple M2 Max')

    def test_older_os_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.supported_host('Darwin', 'arm64', '15.0', 'Apple M4')

    def test_source_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            alias = root / 'alias'
            alias.symlink_to(source, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.new_output_path(alias / 'new-build', source, root / 'runtime')

    def test_runtime_alias_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / 'runtime'
            runtime.mkdir()
            alias = root / 'alias'
            alias.symlink_to(runtime, target_is_directory=True)
            with self.assertRaises(ValueError):
                MODULE.new_output_path(alias / 'new-build', root / 'source', runtime)

    def test_external_new_output_resolves_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / 'safe'
            safe.mkdir()
            alias = root / 'alias'
            alias.symlink_to(safe, target_is_directory=True)
            result = MODULE.new_output_path(alias / 'new-build', root / 'source', root / 'runtime')
            self.assertEqual(result, safe.resolve() / 'new-build')
            self.assertFalse(result.exists())

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                MODULE.new_output_path(root, root / 'source', root / 'runtime')

    def test_exact_config(self):
        MODULE.config_check('set(GGML_VERSION "0.10.0")\n', {'GGML_VERSION': '0.10.0'})

    def test_wrong_config_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.config_check('set(GGML_VERSION "0.11.0")\n', {'GGML_VERSION': '0.10.0'})

    def test_missing_config_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.config_check('', {'GGML_OPENMP': 'ON'})


if __name__ == '__main__':
    unittest.main()
