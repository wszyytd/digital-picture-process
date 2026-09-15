"""Shared standard-library helpers; runnable without a ML environment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve(path):
    p = Path(path).expanduser()
    return (p if p.is_absolute() else ROOT / p).resolve()


def model_python(model):
    env = ROOT / ".venvs" / model
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def process_env():
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               UV_CACHE_DIR=str(ROOT / ".cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(ROOT / ".python"),
               YOLO_CONFIG_DIR=str(ROOT / ".cache/ultralytics"),
               YOLOV5_CONFIG_DIR=str(ROOT / ".cache/yolov5"),
               TORCH_HOME=str(ROOT / ".cache/torch"),
               MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
               YOLOv5_AUTOINSTALL="false", WANDB_MODE="disabled")
    return env


def git_args(*args):
    # Work around Windows Schannel failures without disabling TLS verification.
    return ["git", *(["-c", "http.sslBackend=openssl"] if os.name == "nt" else []), *args]


def capture(command, cwd=ROOT):
    return subprocess.check_output(command, cwd=cwd, env=process_env(),
                                   text=True, encoding="utf-8", errors="replace").strip()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        print("Downloading:", url, flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "yolo-workspace/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        partial.replace(target)
    return {"url": url, "file": target.name, "sha256": sha256(target)}


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or ":" in name:
                raise ValueError("Unsafe archive member: " + member.filename)
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
        z.extractall(destination)


def verify_upstream(model):
    spec = load_json(ROOT / "configs/models.json")[model]
    if "repository" not in spec:
        return
    repo = ROOT / "third_party" / spec["directory"]
    if not (repo / ".git").exists():
        raise ValueError("Missing upstream source; run scripts/setup.py --model " + model)
    head = capture(git_args("rev-parse", "HEAD"), repo)
    if head != spec["commit"]:
        raise ValueError("Upstream revision differs from configs/models.json: " + str(repo))
    if capture(git_args("status", "--porcelain", "--untracked-files=no"), repo):
        raise ValueError("Upstream source has tracked modifications: " + str(repo))
