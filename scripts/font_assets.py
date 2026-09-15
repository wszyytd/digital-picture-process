"""Prepare plotting fonts locally; never download optional fonts at training time."""
from pathlib import Path
import shutil


def ensure_plot_font(cache_dir, candidates):
    """Cache a local TrueType font under the filename required by YOLOv5.

    Arial is preferred. DejaVu Sans is an offline fallback for our ASCII class
    labels, not a replacement for Arial Unicode when using CJK class names.
    Fonts remain in the ignored machine-local cache, never in the repository.
    """
    target = Path(cache_dir) / "Arial.ttf"
    if target.is_file() and target.stat().st_size:
        return target
    for candidate in candidates:
        source = Path(candidate)
        if source.is_file() and source.stat().st_size:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            print("Plot font:", source, "->", target, flush=True)
            return target
    raise FileNotFoundError("No local plot font found. Install the model's matplotlib dependency or supply Arial.ttf in " + str(cache_dir))
