"""CLI contracts for the project's real external boundaries, without torch."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class WorkspaceTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/run.py'), *args],
                              capture_output=True, text=True, encoding='utf-8')

    def test_cpu_dry_run_for_all_models(self):
        for model in ('yolov5', 'yolov7-tiny', 'yolov8'):
            with self.subTest(model=model):
                result = self.cli('--model', model, '--profile', 'cpu-smoke', '--dry-run')
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertIn('cpu', ' '.join(plan['command']))
                self.assertIn('.venvs', plan['command'][0])
                self.assertTrue(Path(plan['cwd']).is_absolute())
                if model == 'yolov7-tiny':
                    self.assertTrue(any('cfg/training/yolov7-tiny.yaml' in x for x in plan['command']))
                    self.assertTrue(any('hyp.scratch.tiny.yaml' in x for x in plan['command']))

    def test_unicode_data_root_remains_one_argument(self):
        result = self.cli('--model', 'yolov5', '--profile', 'server',
                          '--data-root', str(ROOT / '测试 数据'), '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertIn('测试 数据', plan['dataset']['train'])
        self.assertIn('data.yaml', plan['command'][plan['command'].index('--data') + 1])

    def test_name_cannot_escape_runs_directory(self):
        result = self.cli('--model', 'yolov5', '--name', '../../outside', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('name', result.stderr)

    def test_validation_requires_explicit_weights(self):
        result = self.cli('--model', 'yolov8', '--action', 'val', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('weights', result.stderr)

    def test_prediction_requires_explicit_source(self):
        result = self.cli('--model', 'yolov5', '--action', 'predict',
                          '--weights', 'weights/yolov5n.pt', '--dry-run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('source', result.stderr)

    def test_v5_validation_forwards_cpu_workers(self):
        result = self.cli('--model', 'yolov5', '--action', 'val',
                          '--weights', 'weights/yolov5n.pt', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(result.stdout)['command']
        self.assertIn('--workers', command)
        self.assertEqual(command[command.index('--workers') + 1], '0')

    def test_v7_validation_carries_explicit_loader_worker_limit(self):
        result = self.cli('--model', 'yolov7-tiny', '--action', 'val',
                          '--weights', 'weights/yolov7-tiny.pt', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan.get('runtime_env', {}).get('YOLO_WORKERS'), '0')

    def test_zip_traversal_rejected(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        try:
            from workspace import extract_zip
        except ImportError:
            self.fail('workspace.extract_zip not implemented')
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            archive = base / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside.txt', 'bad')
            with self.assertRaises(ValueError):
                extract_zip(archive, base / 'out')
            self.assertFalse((base / 'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()
