#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

import torch


def configure_torch() -> None:
    torch.backends.cudnn.enabled = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)


def main() -> None:
    configure_torch()
    eval_server_path = Path(__file__).resolve().parents[1] / "robometer" / "evals" / "eval_server.py"
    runpy.run_path(str(eval_server_path), run_name="__main__")


if __name__ == "__main__":
    main()
