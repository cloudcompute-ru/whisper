import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProvisionRuntimeTest(unittest.TestCase):
    def run_install(self, existing=False, fail=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            python_stub = root / 'python-stub'
            python_stub.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ['TEST_ROOT'])
with (root / 'calls.jsonl').open('a') as f:
    f.write(json.dumps(args) + '\\n')
if args == ['-m', 'pip', '--version']:
    sys.exit(0 if (root / 'pip-ready').exists() else 1)
if args[:2] == ['-m', 'ensurepip']:
    (root / 'pip-ready').touch()
if args[:1] == ['-c']:
    assert 'nvidia.cublas.lib' in args[1] and 'nvidia.cudnn.lib' in args[1]
    print('/venv/cublas/lib:/venv/cudnn/lib')
if args[:3] == ['-m', 'pip', 'install']:
    assert (root / 'pip-ready').exists(), 'pip used before installation'
    if 'torch' in args:
        assert args[args.index('--index-url') + 1] == 'https://download.pytorch.org/whl/cu128'
        assert not any('gradio' in a or 'faster-whisper' in a for a in args)
    if 'faster-whisper==1.1.1' in args:
        assert args[args.index('--index-url') + 1] == 'https://pypi.org/simple'
        assert 'requests>=2.32,<3' in args
        if os.environ.get('FAIL_INSTALL') == '1':
            sys.exit(19)
''')
            python_stub.chmod(0o755)
            uv = bin_dir / 'uv'
            uv.write_text('''#!/usr/bin/env python3
import os, pathlib, shutil, sys
assert '--seed' in sys.argv
root = pathlib.Path(os.environ['TEST_ROOT'])
dest = pathlib.Path(sys.argv[-1]) / 'bin'
dest.mkdir(parents=True)
shutil.copy(root / 'python-stub', dest / 'python')
(root / 'pip-ready').touch()
''')
            uv.chmod(0o755)
            curl = bin_dir / 'curl'
            curl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
args = sys.argv[1:]
if '-d' in args:
    with (pathlib.Path(os.environ['TEST_ROOT']) / 'reports.jsonl').open('a') as f:
        f.write(args[args.index('-d') + 1] + '\\n')
''')
            curl.chmod(0o755)
            app = root / 'app'
            if existing:
                dest = app / 'venv/bin'
                dest.mkdir(parents=True)
                (dest / 'python').symlink_to(python_stub)
            script = root / 'install.sh'
            script.write_text((ROOT / 'provision.sh').read_text().split('# --- download_model')[0]
                              + '\necho "CUDA_LIBRARIES=$LD_LIBRARY_PATH"\n')
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       TEST_ROOT=str(root), APP_DIR=str(app), FAIL_INSTALL=str(int(fail)),
                       CC_PROVISION_URL='https://example.invalid/provision', CC_AGENT_TOKEN='test-token',
                       LD_LIBRARY_PATH='/existing/cuda/lib')
            result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            reports = [json.loads(line) for line in (root / 'reports.jsonl').read_text().splitlines()]
            return result, calls, reports

    def test_clean_environment_installs_packages_from_their_correct_indexes(self):
        result, calls, reports = self.run_install()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(3, sum(c[:3] == ['-m', 'pip', 'install'] for c in calls))
        self.assertTrue(any(r.get('progress_pct') == 100 for r in reports))
        self.assertIn('CUDA_LIBRARIES=/venv/cublas/lib:/venv/cudnn/lib:/existing/cuda/lib', result.stdout)

    def test_existing_environment_without_pip_is_recovered(self):
        result, calls, _ = self.run_install(existing=True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(['-m', 'ensurepip', '--upgrade'], calls)

    def test_install_failure_reports_error_and_preserves_nonzero_exit(self):
        result, _, reports = self.run_install(fail=True)
        self.assertEqual(19, result.returncode)
        self.assertEqual('install_runtime', reports[-1]['stage'])
        self.assertIn('exit 19', reports[-1]['message'])
        self.assertFalse(any(r.get('progress_pct') == 100 for r in reports))


if __name__ == '__main__':
    unittest.main()
