"""Launch official YOLO implementations with shared dataset and runtime profiles."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from workspace import ROOT, capture, git_args, load_json, model_python, process_env, resolve, save_json, sha256, verify_upstream


def build_plan(args):
    models = load_json(ROOT / "configs/models.json")
    spec = models[args.model]
    profile = load_json(ROOT / "configs/profiles" / (args.profile + ".json"))
    for key in ("device", "epochs", "imgsz", "batch", "workers"):
        if getattr(args, key) is not None:
            profile[key] = getattr(args, key)
    if os.name == "nt" and args.model == "yolov7-tiny":
        profile["workers"] = 0  # Spawned Windows workers would bypass Unicode I/O adaptation.
    for key in ("epochs", "imgsz", "batch"):
        if profile[key] < 1:
            raise ValueError(key + " must be positive")
    if profile["imgsz"] % 32:
        raise ValueError("imgsz must be a multiple of 32")
    if profile["workers"] < 0:
        raise ValueError("workers cannot be negative")
    name = args.name or (args.action + "-" + args.profile + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("name must contain only ASCII letters, digits, '-' and '_'")
    if args.action != "train" and not args.weights:
        raise ValueError("--weights is required for val/predict")
    if args.action == "predict" and not args.source:
        raise ValueError("--source is required for predict")
    manifest_path = resolve(args.dataset or profile["dataset"])
    manifest = load_json(manifest_path)
    names = manifest["names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("Dataset names must be a nonempty list of unique class names")
    data_root = resolve(args.data_root or manifest["root"])
    dataset = {"nc": len(names), "names": names}
    for split in ("train", "val", "test"):
        if split in manifest:
            dataset[split] = (data_root / manifest[split]).resolve().as_posix()
    for split in ("train", "val"):
        if split not in dataset:
            raise ValueError("Dataset requires train and val")
    if args.action == "val" and args.split not in dataset:
        raise ValueError("Dataset has no " + args.split + " split")
    run_dir = ROOT / "runs" / args.model / name
    dataset_file = run_dir / "data.yaml"
    weights = resolve(args.weights) if args.weights else ROOT / "weights" / spec["weights"]
    python = model_python(args.model)
    command = [str(python), "-X", "utf8"]
    common = {"device": str(profile["device"]), "imgsz": str(profile["imgsz"]),
              "batch": str(profile["batch"]), "workers": str(profile["workers"])}
    cwd = ROOT
    if args.model == "yolov8":
        command += ["-c", "from ultralytics.cfg import entrypoint; entrypoint()", "detect", args.action,
                    "model=" + str(weights), "project=" + str(run_dir.parent), "name=" + name,
                    "exist_ok=True", "device=" + common["device"], "imgsz=" + common["imgsz"]]
        if args.action == "train":
            command += ["data=" + str(dataset_file), "epochs=" + str(profile["epochs"]),
                        "batch=" + common["batch"], "workers=" + common["workers"],
                        "seed=" + str(profile["seed"]), "optimizer=SGD", "amp=False", "patience=0"]
        elif args.action == "val":
            command += ["data=" + str(dataset_file), "split=" + args.split, "batch=" + common["batch"],
                        "workers=" + common["workers"], "conf=0.001", "iou=0.6", "max_det=300"]
        else:
            command += ["source=" + str(resolve(args.source)), "save=True", "conf=0.25"]
    else:
        cwd = ROOT / "third_party" / spec["directory"]
        script = "train.py" if args.action == "train" else ("detect.py" if args.action == "predict" else ("val.py" if args.model == "yolov5" else "test.py"))
        command += [str(ROOT / "scripts/legacy_entry.py"), str(cwd / script),
                    "--weights", str(weights), "--project", str(run_dir.parent), "--name", name,
                    "--exist-ok", "--device", common["device"]]
        if args.action == "train":
            command += ["--data", str(dataset_file), "--epochs", str(profile["epochs"]),
                        "--batch-size", common["batch"], "--workers", common["workers"]]
            if args.model == "yolov5":
                command += ["--imgsz", common["imgsz"], "--seed", str(profile["seed"]), "--patience", "0"]
            else:
                command += ["--img-size", common["imgsz"], common["imgsz"],
                            "--cfg", (cwd / "cfg/training/yolov7-tiny.yaml").as_posix(),
                            "--hyp", (cwd / "data/hyp.scratch.tiny.yaml").as_posix(), "--v5-metric"]
        elif args.action == "val":
            command += ["--data", str(dataset_file), "--task", args.split,
                        "--batch-size", common["batch"], "--img-size", common["imgsz"],
                        "--conf-thres", "0.001", "--iou-thres", "0.6"]
            if args.model == "yolov7-tiny":
                command += ["--v5-metric"]
            else:
                command += ["--workers", common["workers"]]
        else:
            command += ["--source", str(resolve(args.source)), "--img-size", common["imgsz"], "--conf-thres", "0.25"]
    return {"model": args.model, "action": args.action, "profile": profile, "upstream": spec,
            "command": command, "cwd": str(cwd), "run_dir": str(run_dir),
            "runtime_env": {"YOLO_WORKERS": str(profile["workers"]),
                            "OMP_NUM_THREADS": str(profile["threads"]),
                            "MKL_NUM_THREADS": str(profile["threads"])},
            "dataset": dataset, "dataset_manifest": str(manifest_path), "weights": str(weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=load_json(ROOT / "configs/models.json"), required=True)
    parser.add_argument("--profile", choices=["cpu-smoke", "server"], default="cpu-smoke")
    parser.add_argument("--action", choices=["train", "val", "predict"], default="train")
    parser.add_argument("--dataset", help="Project dataset JSON manifest")
    parser.add_argument("--data-root", help="Dataset root override for this machine")
    parser.add_argument("--weights", help="Explicit checkpoint path for val/predict or training")
    parser.add_argument("--source", help="Image/video file or directory for predict")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--name", help="Unique experiment name; existing directories are refused")
    parser.add_argument("--device")
    for key in ("epochs", "imgsz", "batch", "workers"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloads or writes")
    args = parser.parse_args()
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        python = model_python(args.model)
        if not python.exists():
            raise ValueError("Missing model environment. Run scripts/setup.py --model " + args.model)
        verify_upstream(args.model)
        if not Path(plan["weights"]).is_file():
            raise ValueError("Missing weights: " + plan["weights"])
        if args.action == "predict":
            if not resolve(args.source).exists():
                raise ValueError("Missing source: " + args.source)
        else:
            required = ("train", "val") if args.action == "train" else (args.split,)
            for split in required:
                if not Path(plan["dataset"][split]).exists():
                    raise ValueError("Missing data split: " + plan["dataset"][split] + "; prepare the dataset first")
        if plan["profile"]["device"] != "cpu":
            probe = capture([str(python), "-c", "import torch; print(torch.cuda.is_available())"])
            if probe != "True":
                raise ValueError("CUDA unavailable. Use --device cpu or install a compatible GPU runtime")
        run_dir = Path(plan["run_dir"])
        if run_dir.exists():
            raise ValueError("Experiment already exists; use a new --name: " + str(run_dir))
        run_dir.mkdir(parents=True)
        # JSON is a YAML subset understood by all three pinned PyYAML consumers.
        save_json(run_dir / "data.yaml", plan["dataset"])
        env = process_env()
        env.update(plan["runtime_env"])
        for directory in (".cache/yolov5", ".cache/ultralytics", ".cache/matplotlib", ".cache/torch"):
            (ROOT / directory).mkdir(parents=True, exist_ok=True)
        plan["started_at"] = datetime.now(timezone.utc).isoformat()
        plan["weights_sha256"] = sha256(plan["weights"])
        plan["status"] = "running"
        try:
            plan["workspace_commit"] = capture(git_args("rev-parse", "HEAD"))
        except subprocess.CalledProcessError:
            plan["workspace_commit"] = None
        plan["workspace_dirty"] = bool(capture(git_args("status", "--porcelain")))
        inventory = capture([str(python), "-c",
                             "import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions())))"])
        (run_dir / "environment.txt").write_text(inventory + "\n", encoding="utf-8")
        save_json(run_dir / "experiment.json", plan)
        print("Experiment:", run_dir, flush=True)
        print("SMOKE ONLY" if args.profile == "cpu-smoke" else "Training/evaluation profile: server", flush=True)
        returncode = 1
        try:
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(plan["command"], cwd=plan["cwd"], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
                    try:
                        for line in proc.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        returncode = proc.wait()
                    except KeyboardInterrupt:
                        proc.terminate()
                        proc.wait()
                        returncode = 130
        finally:
            plan["exit_code"] = returncode
            plan["status"] = "completed" if returncode == 0 else "failed"
            plan["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_json(run_dir / "experiment.json", plan)
        return returncode
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
