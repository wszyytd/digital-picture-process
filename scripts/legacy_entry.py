"""Run an official legacy script, adding Windows Unicode image I/O support."""
from pathlib import Path
import inspect
import os
import runpy
import sys


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    if (script.parent / "models/yolov5n.yaml").exists():
        import matplotlib
        from font_assets import ensure_plot_font
        from workspace import ROOT

        cache = Path(os.environ.get("YOLOV5_CONFIG_DIR", ROOT / ".cache/yolov5"))
        os.environ["YOLOV5_CONFIG_DIR"] = str(cache)
        windows_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf"
        fallback = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        ensure_plot_font(cache, [windows_font, fallback])
    # OpenCV's Windows imread/imwrite do not reliably accept Unicode file names.
    if sys.platform == "win32":
        import cv2
        import numpy as np

        def imread(filename, flags=cv2.IMREAD_COLOR):
            try:
                return cv2.imdecode(np.fromfile(str(filename), dtype=np.uint8), flags)
            except (OSError, cv2.error):
                return None

        def imwrite(filename, image, params=None):
            try:
                success, buffer = cv2.imencode(Path(filename).suffix, image, params or [])
                if success:
                    buffer.tofile(str(filename))
                return success
            except (OSError, cv2.error):
                return False

        cv2.imread = imread
        cv2.imwrite = imwrite
    # YOLOv7 test.py has no --workers option. Patch the common loader before
    # train.py/test.py import it, so both train and standalone val obey the profile.
    if (script.parent / "cfg/training/yolov7-tiny.yaml").exists():
        import utils.datasets as datasets
        original_loader = datasets.create_dataloader
        signature = inspect.signature(original_loader)
        workers = int(os.environ.get("YOLO_WORKERS", "0"))

        def create_dataloader(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments["workers"] = workers
            return original_loader(*bound.args, **bound.kwargs)

        datasets.create_dataloader = create_dataloader
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
