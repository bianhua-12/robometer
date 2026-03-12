#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import uvicorn

from robometer.configs.eval_configs import EvalServerConfig
from robometer.evals import eval_server
from robometer.utils import save as save_utils
from robometer.utils import setup_utils


def configure_torch() -> None:
    # Leave optimized attention backends enabled for the non-Unsloth path.
    torch.backends.cudnn.enabled = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(True)


def make_no_unsloth_wrapper(base_model_override: str | None):
    original = setup_utils.setup_model_and_processor

    def wrapper(cfg: Any, hf_model_id: str = "", peft_config: Any = None):
        patched_cfg = replace(cfg, use_unsloth=False)
        if base_model_override:
            patched_cfg = replace(patched_cfg, base_model_id=base_model_override)
        eval_server.logger.info(
            f"No-Unsloth launcher override: use_unsloth={patched_cfg.use_unsloth} "
            f"base_model_id={patched_cfg.base_model_id}"
        )
        return original(patched_cfg, hf_model_id, peft_config)

    return wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval_server with standard transformers loading.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-url", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8002)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--base-model-id", default=None)
    parser.add_argument(
        "--dry-run-load",
        action="store_true",
        help="Load the model on CPU once and exit without serving.",
    )
    parser.add_argument(
        "--load-only-cpu",
        action="store_true",
        help="Resolve checkpoint and load the model on CPU without starting the server.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch()

    if args.base_model_id:
        base_model_id = str(Path(args.base_model_id).expanduser().resolve())
    else:
        base_model_id = None

    setup_utils.setup_model_and_processor = make_no_unsloth_wrapper(base_model_id)

    cfg = EvalServerConfig(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        server_url=args.server_url,
        server_port=args.server_port,
        num_gpus=args.num_gpus,
        max_workers=args.max_workers,
    )

    if args.load_only_cpu:
        exp_config, _, _, model = save_utils.load_model_from_hf(cfg.model_path, device=torch.device("cpu"))
        print(f"Loaded model on CPU: {model.__class__.__name__}")
        print(f"use_unsloth={exp_config.model.use_unsloth}")
        print(f"base_model_id={exp_config.model.base_model_id}")
        return

    if args.dry_run_load:
        server = eval_server.MultiGPUEvalServer(cfg.model_path, cfg.num_gpus, cfg.max_workers)
        server.shutdown()
        return

    server = eval_server.MultiGPUEvalServer(cfg.model_path, cfg.num_gpus, cfg.max_workers)
    app = eval_server.create_app(cfg, server)
    print(f"Running no-unsloth eval server on {cfg.server_url}:{cfg.server_port}")
    print(f"Using {cfg.num_gpus} GPUs")
    if base_model_id:
        print(f"Base model override: {base_model_id}")
    uvicorn.run(app, host=cfg.server_url, port=cfg.server_port)


if __name__ == "__main__":
    main()
