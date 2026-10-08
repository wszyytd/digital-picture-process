"""Short commands for YOLOv8n on the cleaned 14-class ODSR dataset.

All model execution and experiment recording remain in setup.py/run.py.
Relative paths are interpreted relative to the project root.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup', 'check', 'train', 'test'])
    parser.add_argument('--backend', choices=['cpu', 'cu118', 'cu121', 'cu124'], default='cu121', help='setup only')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', default='0')
    parser.add_argument('--name')
    parser.add_argument('--weights', help='Required for test; optional initialization for training')
    parser.add_argument('--data-root', help='Optional dataset copy; default data/odsr-ihs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-X', 'utf8']
    if args.mode == 'setup':
        command += [str(ROOT / 'scripts/setup.py'), '--model', 'yolov8', '--backend', args.backend]
        if args.dry_run:
            print(json.dumps({'command': command, 'cwd': str(ROOT)}, indent=2))
            return 0
    else:
        if args.mode == 'test' and not args.weights:
            parser.error('test requires --weights pointing to the selected best.pt')
        epochs = args.epochs if args.epochs is not None else (3 if args.mode == 'check' else 50)
        workers = args.workers if args.workers is not None else (0 if args.mode == 'test' else 4)
        names = {'check': 'v8n_odsr_check_01', 'train': f'v8n_odsr_b{args.batch}_e{epochs}_01', 'test': 'v8n_odsr_test_01'}
        command += [str(ROOT / 'scripts/run.py'), '--model', 'yolov8', '--profile', 'server',
                    '--dataset', 'configs/datasets/odsr-ihs.json',
                    '--batch', str(args.batch), '--imgsz', str(args.imgsz),
                    '--workers', str(workers), '--device', args.device,
                    '--name', args.name or names[args.mode]]
        if args.mode == 'test':
            command += ['--action', 'val', '--split', 'test']
        else:
            command += ['--epochs', str(epochs)]
        if args.weights:
            command += ['--weights', args.weights]
        if args.data_root:
            command += ['--data-root', args.data_root]
        if args.dry_run:
            command += ['--dry-run']
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
