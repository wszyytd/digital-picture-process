# YOLO Workspace Implementation Plan

> For agentic workers: execute sequentially in the user-authorized workspace. Follow verification-before-completion and request a focused code review before delivery.

**Goal:** Create reproducible YOLOv5, YOLOv7-tiny and YOLOv8 entry points for Windows CPU validation and Linux GPU training.

**Architecture:** Standard-library Python orchestration, independent model environments, shared dataset manifests and fixed upstream revisions. Training artifacts stay outside Git.

**Tech Stack:** Python 3.10 model environments, uv, Git, PyTorch, three official YOLO implementations.

**Spec:** docs/superpowers/specs/2026-09-14-yolo-workspace-design.md

## Global Constraints

- Work directly in D:/Work/数字图像处理 as explicitly requested.
- Preserve init. No GitHub publishing or server access in this step.
- No machine-specific paths in committed experiment manifests.
- Smoke metrics must never be presented as formal evaluation.

## Tasks

- [x] Create shared model/dataset/profile manifests and ignored artifact directories.
- [x] Write CLI contract tests: CPU routing, v7 tiny architecture, Unicode paths, invalid names, missing data and safe zip extraction. Run unittest before implementation.
- [x] Implement setup.py with per-model uv environments, fixed upstream checkout and official pretrained downloads.
- [x] Implement prepare_smoke.py and run.py. Use subprocess argument lists; resolve dataset paths per machine; record experiment metadata.
- [x] Add Chinese README, local/server instructions, data contract and comparison protocol.
- [x] Run python -m unittest discover -s tests -v, all model/action dry-runs and git diff --check. Attempt v5 CPU smoke where environment installation succeeds.
- [x] Review script behavior and Git exclusions; commit only intended source/config/documentation files and report exact verification limits.

实际模型训练因官方下载 TLS 失败未执行；任务中的“尝试 CPU smoke”已尝试并记录阻断。离线检查与结构初始化已完成。
