"""Exercise the V8 shortcut through the real run.py planner, without torch."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class V8EntryTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/train_v8.py'), *args], cwd=ROOT.parent, capture_output=True, text=True, encoding='utf-8')

    def plan(self, *args):
        result = self.cli(*args, '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_check_and_train_start_from_official_weights(self):
        for mode, epochs in [('check', 3), ('train', 50)]:
            p = self.plan(mode)
            self.assertEqual(p['model'], 'yolov8')
            self.assertEqual(p['dataset']['nc'], 14)
            self.assertEqual(Path(p['weights']).name, 'yolov8n.pt')
            self.assertIn('epochs=' + str(epochs), p['command'])
            self.assertIn('batch=32', p['command'])
            self.assertIn('imgsz=640', p['command'])
            self.assertIn('odsr-ihs/images/train', p['dataset']['train'])

    def test_overrides_reach_real_planner_from_other_directory(self):
        p = self.plan('train', '--batch', '16', '--epochs', '70', '--device', 'cpu', '--workers', '0', '--name', 'custom_v8')
        for arg in ['batch=16', 'epochs=70', 'device=cpu', 'workers=0', 'name=custom_v8']:
            self.assertIn(arg, p['command'])

    def test_test_requires_explicit_checkpoint(self):
        r = self.cli('test', '--dry-run')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--weights', r.stderr)

    def test_test_uses_test_split_and_given_checkpoint(self):
        p = self.plan('test', '--weights', 'runs/yolov8/example/weights/best.pt')
        self.assertEqual(p['action'], 'val')
        for arg in ['split=test', 'conf=0.001', 'iou=0.6', 'max_det=300', 'workers=0']:
            self.assertIn(arg, p['command'])
        self.assertTrue(p['weights'].endswith('best.pt'))

    def test_invalid_batch_is_not_silently_accepted(self):
        r = self.cli('train', '--batch', '0', '--dry-run')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('batch must be positive', r.stderr)

if __name__ == '__main__':
    unittest.main()
