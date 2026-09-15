"""Prepare one model environment. Does not modify the system Python."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from workspace import ROOT, capture, download, git_args, load_json, model_python, process_env, save_json, verify_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = load_json(ROOT / "configs/models.json")
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--backend", choices=["cpu", "cu118", "cu121", "cu124"], default="cpu")
    parser.add_argument("--python", default="3.10", help="uv Python selector; defaults to Python 3.10")
    parser.add_argument("--assets-only", action="store_true", help="Only fetch source and weights")
    args = parser.parse_args()
    spec = models[args.model]
    env = process_env()
    if "repository" in spec:
        repo = ROOT / "third_party" / spec["directory"]
        if not repo.exists():
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(git_args("init", str(repo)), check=True, env=env)
            subprocess.run(git_args("-C", str(repo), "remote", "add", "origin", spec["repository"]), check=True, env=env)
            subprocess.run(git_args("-C", str(repo), "fetch", "--depth", "1", "origin", spec["commit"]), check=True, env=env)
            subprocess.run(git_args("-C", str(repo), "checkout", "--detach", "FETCH_HEAD"), check=True, env=env)
        elif not (repo / "train.py").exists() and (repo / ".git").exists():
            # Resume an interrupted first download, without overwriting a working checkout.
            subprocess.run(git_args("-C", str(repo), "fetch", "--depth", "1", spec["repository"], spec["commit"]), check=True, env=env)
            subprocess.run(git_args("-C", str(repo), "checkout", "--detach", "FETCH_HEAD"), check=True, env=env)
        verify_upstream(args.model)
    record = download(spec["weights_url"], ROOT / "weights" / spec["weights"])
    save_json(ROOT / "weights" / (spec["weights"] + ".source.json"), record)
    if args.assets_only:
        return
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("uv is required. Install uv first: https://docs.astral.sh/uv/getting-started/installation/")
    python = model_python(args.model)
    if not python.exists():
        subprocess.run([uv, "venv", "--python", args.python, str(ROOT / ".venvs" / args.model)],
                       check=True, env=env)
    version = capture([str(python), "-c", "import sys;print('.'.join(map(str,sys.version_info[:2])))"])
    if version != "3.10":
        raise ValueError("The pinned dependencies require a Python 3.10 environment; got " + version)
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    "torch==2.5.1+" + args.backend, "torchvision==0.20.1+" + args.backend,
                    "--index-url", "https://download.pytorch.org/whl/" + args.backend], check=True, env=env)
    subprocess.run([uv, "pip", "install", "--python", str(python),
                    "-r", str(ROOT / "environments" / (args.model + ".txt"))], check=True, env=env)
    subprocess.run([uv, "pip", "check", "--python", str(python)], check=True, env=env)
    actual_build = capture([str(python), "-c", "import torch; print(torch.__version__)"])
    if actual_build != "2.5.1+" + args.backend:
        raise ValueError("Installed torch build does not match selected backend: " + actual_build)
    installed = capture([uv, "pip", "freeze", "--python", str(python)])
    lock = ROOT / ".venvs" / args.model / "requirements-resolved.txt"
    lock.write_text(installed + "\n", encoding="utf-8")
    save_json(ROOT / ".venvs" / args.model / "setup.json",
              {"model": args.model, "backend": args.backend, "python": version, "upstream": spec})
    subprocess.run([str(python), "-c",
                    "import torch; print('torch:',torch.__version__); print('CUDA available:',torch.cuda.is_available())"],
                   check=True, env=env)
    print("Ready:", python)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
