#!/usr/bin/env python3
"""Build a Robometer dense-reward advantage parquet from source episode parquet files."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from PIL import Image

from scripts.example_inference import build_multipart_payload, make_progress_sample


DEFAULT_SOURCE_ROOT = Path(
    "/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot_collect_512_task0_fail30/collected_data_3shot_512_task0_merged"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / "data"
    / "libero_10_3shot_collect_512_task0_fail30"
    / "collected_data_3shot_512_task0_merged"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Robometer prefix-step inference on source parquet episodes and write an advantage parquet."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chunk-size", type=int, default=48)
    parser.add_argument("--parallel-chunks", type=int, default=3)
    parser.add_argument("--prefetch-episodes", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def decode_frames(image_cells: list[dict[str, object]]) -> np.ndarray:
    frames = []
    for cell in image_cells:
        image_bytes = cell.get("bytes")
        if image_bytes is None:
            raise ValueError("Encountered image cell without embedded bytes.")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames, axis=0)


def select_prefix_endpoints(num_frames: int, frame_stride: int) -> list[int]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    endpoints = list(range(1, num_frames + 1, frame_stride))
    if not endpoints or endpoints[-1] != num_frames:
        endpoints.append(num_frames)
    return endpoints


def expand_prefix_samples(frames: np.ndarray, task: str, prefix_endpoints: list[int]) -> list[dict]:
    samples: list[dict] = []
    for i in prefix_endpoints:
        indices = np.linspace(0, i - 1, 4, dtype=int)
        sub_frames = frames[indices]
        samples.append(
            make_progress_sample(
                frames=sub_frames,
                task=task,
                sample_id=str(i),
                subsequence_length=i,
            )
        )
    return samples


def build_prefix_samples_for_range(
    frames: np.ndarray,
    task: str,
    prefix_endpoints: list[int],
    start: int,
    end: int,
) -> list[dict]:
    return expand_prefix_samples(frames, task, prefix_endpoints[start:end])


def post_batch(url: str, samples: list[dict], timeout_s: float) -> dict:
    payload_start = time.perf_counter()
    files, data = build_multipart_payload(samples)
    payload_s = time.perf_counter() - payload_start
    request_start = time.perf_counter()
    resp = requests.post(
        url.rstrip("/") + "/evaluate_batch_npy",
        files=files,
        data=data,
        timeout=timeout_s,
    )
    request_s = time.perf_counter() - request_start
    resp.raise_for_status()
    decode_start = time.perf_counter()
    outputs = resp.json()
    decode_s = time.perf_counter() - decode_start
    return {
        "outputs": outputs,
        "payload_s": payload_s,
        "request_s": request_s,
        "response_decode_s": decode_s,
    }


@dataclass
class EpisodeState:
    stem: str
    parquet_path: Path
    index: int
    total_episodes: int
    episode_df: pd.DataFrame
    frames: np.ndarray
    task: str
    prefix_endpoints: list[int]
    sampled_rewards: np.ndarray
    sampled_success_probs: np.ndarray
    total_chunks: int
    dataset_name: str
    frame_stride: int
    load_parquet_s: float
    decode_frames_s: float
    prepare_prefix_s: float
    episode_start_t: float
    next_chunk_start: int = 0
    chunks_completed: int = 0
    chunk_samples_completed: int = 0
    inference_start_t: float = 0.0


def collect_prefix_outputs(
    eval_server_url: str,
    samples: list[dict],
    chunk_size: int,
    timeout_s: float,
    parallel_chunks: int,
    retries: int,
) -> tuple[np.ndarray, np.ndarray]:
    chunks = [(start, samples[start : start + chunk_size]) for start in range(0, len(samples), chunk_size)]
    rewards = np.zeros(len(samples), dtype=np.float32)
    success = np.zeros(len(samples), dtype=np.float32)
    total_chunks = len(chunks)
    collect_start = time.perf_counter()

    def run_chunk(item: tuple[int, list[dict]]) -> tuple[int, list[float], list[float], dict[str, float]]:
        start, chunk = item
        attempt = 0
        while True:
            chunk_start = time.perf_counter()
            try:
                result = post_batch(eval_server_url, chunk, timeout_s)
                outputs = result["outputs"]
                progress_pred = outputs.get("outputs_progress", {}).get("progress_pred", [])
                success_probs = outputs.get("outputs_success", {}).get("success_probs", [])
                chunk_rewards = [float(pred[-1]) for pred in progress_pred]
                chunk_success = [float(pred[-1]) for pred in success_probs]
                if len(chunk_success) < len(chunk_rewards):
                    chunk_success.extend([0.0] * (len(chunk_rewards) - len(chunk_success)))
                timings = {
                    "chunk_total_s": time.perf_counter() - chunk_start,
                    "payload_s": float(result["payload_s"]),
                    "request_s": float(result["request_s"]),
                    "response_decode_s": float(result["response_decode_s"]),
                }
                return start, chunk_rewards, chunk_success, timings
            except Exception:
                if attempt >= retries:
                    raise
                attempt += 1
                sleep_s = 5 * attempt
                print(f"retry chunk {start}:{start + len(chunk)} attempt {attempt}/{retries} after {sleep_s}s", flush=True)
                time.sleep(sleep_s)

    with ThreadPoolExecutor(max_workers=parallel_chunks) as executor:
        futures = [executor.submit(run_chunk, item) for item in chunks]
        for future in as_completed(futures):
            start, chunk_rewards, chunk_success, timings = future.result()
            end = start + len(chunk_rewards)
            rewards[start:end] = chunk_rewards
            success[start:end] = chunk_success
            samples_per_s = len(chunk_rewards) / timings["chunk_total_s"] if timings["chunk_total_s"] > 0 else None
            print(
                json.dumps(
                    {
                        "event": "chunk_done",
                        "range": [start, end],
                        "num_samples": len(chunk_rewards),
                        "total_samples": len(samples),
                        "total_chunks": total_chunks,
                        "chunk_total_s": round(timings["chunk_total_s"], 3),
                        "payload_s": round(timings["payload_s"], 3),
                        "request_s": round(timings["request_s"], 3),
                        "response_decode_s": round(timings["response_decode_s"], 3),
                        "samples_per_s": round(samples_per_s, 3) if samples_per_s is not None else None,
                        "elapsed_s": round(time.perf_counter() - collect_start, 3),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    return rewards, success


def interpolate_prefix_outputs(
    sampled_rewards: np.ndarray,
    sampled_success: np.ndarray,
    prefix_endpoints: list[int],
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_x = np.asarray([endpoint - 1 for endpoint in prefix_endpoints], dtype=np.int64)
    full_x = np.arange(num_frames, dtype=np.int64)

    rewards = np.interp(full_x, sample_x, sampled_rewards).astype(np.float32)
    success = np.interp(full_x, sample_x, sampled_success).astype(np.float32)
    return rewards, success


def load_episode_table(parquet_path: Path) -> pd.DataFrame:
    table = pq.read_table(parquet_path, columns=["image", "prompt", "episode_index", "frame_index", "return"])
    df = table.to_pandas()
    return df


def build_episode_advantage_df(
    episode_df: pd.DataFrame,
    rewards: np.ndarray,
    dataset_name: str,
) -> pd.DataFrame:
    if len(episode_df) != len(rewards):
        raise ValueError(f"Frame count mismatch: parquet={len(episode_df)} rewards={len(rewards)}")

    out_df = pd.DataFrame(
        {
            "episode_index": episode_df["episode_index"].astype("int64"),
            "frame_index": episode_df["frame_index"].astype("int64"),
            "advantage_continuous": np.full(len(episode_df), np.nan, dtype=np.float64),
            "return": episode_df["return"].astype("float64"),
            "value_current": rewards.astype(np.float64),
            "value_next": np.full(len(episode_df), np.nan, dtype=np.float64),
            "reward_sum": np.full(len(episode_df), np.nan, dtype=np.float64),
            "reward_sum_raw": np.full(len(episode_df), np.nan, dtype=np.float64),
            "num_valid_rewards": np.zeros(len(episode_df), dtype=np.int64),
            "dataset_name": pd.Series([dataset_name] * len(episode_df), dtype="object"),
            "advantage": np.zeros(len(episode_df), dtype=bool),
        }
    )
    return out_df


def merge_parts(parts_dir: Path, final_path: Path) -> pd.DataFrame:
    part_paths = sorted(parts_dir.glob("episode_*.parquet"))
    if not part_paths:
        raise ValueError(f"No part parquet files found under {parts_dir}")
    frames = [pd.read_parquet(path) for path in part_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(final_path, index=False)
    return merged


def select_episode_paths(data_root: Path, args: argparse.Namespace) -> list[Path]:
    paths = sorted((data_root / "data").rglob("episode_*.parquet"))
    if args.episode_start is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) >= args.episode_start]
    if args.episode_end is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) <= args.episode_end]
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def load_episode_state(
    parquet_path: Path,
    index: int,
    total: int,
    args: argparse.Namespace,
    dataset_name: str,
) -> EpisodeState:
    episode_start_t = time.perf_counter()
    load_start = time.perf_counter()
    episode_df = load_episode_table(parquet_path)
    load_s = time.perf_counter() - load_start

    decode_start = time.perf_counter()
    frames = decode_frames(episode_df["image"].tolist())
    decode_s = time.perf_counter() - decode_start

    task = str(episode_df["prompt"].iloc[0]).strip()
    prefix_prep_start = time.perf_counter()
    prefix_endpoints = select_prefix_endpoints(int(frames.shape[0]), args.frame_stride)
    prepare_prefix_s = time.perf_counter() - prefix_prep_start

    total_chunks = (len(prefix_endpoints) + args.chunk_size - 1) // args.chunk_size
    state = EpisodeState(
        stem=parquet_path.stem,
        parquet_path=parquet_path,
        index=index,
        total_episodes=total,
        episode_df=episode_df,
        frames=frames,
        task=task,
        prefix_endpoints=prefix_endpoints,
        sampled_rewards=np.zeros(len(prefix_endpoints), dtype=np.float32),
        sampled_success_probs=np.zeros(len(prefix_endpoints), dtype=np.float32),
        total_chunks=total_chunks,
        dataset_name=dataset_name,
        frame_stride=int(args.frame_stride),
        load_parquet_s=load_s,
        decode_frames_s=decode_s,
        prepare_prefix_s=prepare_prefix_s,
        episode_start_t=episode_start_t,
        inference_start_t=time.perf_counter(),
    )
    print(
        json.dumps(
            {
                "event": "episode_start",
                "episode": state.stem,
                "index": index,
                "total_episodes": total,
                "frames": int(frames.shape[0]),
                "sampled_prefixes": int(len(prefix_endpoints)),
                "frame_stride": int(args.frame_stride),
                "chunk_size": int(args.chunk_size),
                "parallel_chunks": int(args.parallel_chunks),
                "prefetch_episodes": int(args.prefetch_episodes),
                "num_chunks": int(total_chunks),
                "load_parquet_s": round(load_s, 3),
                "decode_frames_s": round(decode_s, 3),
                "prepare_prefix_s": round(prepare_prefix_s, 3),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return state


def fill_pending_chunks(
    pending_chunks: deque[tuple[str, int, int]],
    active_states: dict[str, EpisodeState],
    active_order: list[str],
    chunk_size: int,
    target_pending: int,
) -> None:
    while len(pending_chunks) < target_pending:
        progressed = False
        for stem in list(active_order):
            state = active_states.get(stem)
            if state is None:
                continue
            if state.next_chunk_start >= len(state.prefix_endpoints):
                continue
            start = state.next_chunk_start
            end = min(start + chunk_size, len(state.prefix_endpoints))
            pending_chunks.append((stem, start, end))
            state.next_chunk_start = end
            progressed = True
            if len(pending_chunks) >= target_pending:
                break
        if not progressed:
            break


def finalize_episode(state: EpisodeState, part_path: Path) -> None:
    interp_start = time.perf_counter()
    rewards, _success_probs = interpolate_prefix_outputs(
        sampled_rewards=state.sampled_rewards,
        sampled_success=state.sampled_success_probs,
        prefix_endpoints=state.prefix_endpoints,
        num_frames=int(state.frames.shape[0]),
    )
    interp_s = time.perf_counter() - interp_start

    write_start = time.perf_counter()
    out_df = build_episode_advantage_df(state.episode_df, rewards, dataset_name=state.dataset_name)
    out_df.to_parquet(part_path, index=False)
    write_s = time.perf_counter() - write_start

    inference_s = time.perf_counter() - state.inference_start_t
    print(
        json.dumps(
            {
                "episode": state.stem,
                "frames": int(len(out_df)),
                "sampled_prefixes": int(len(state.prefix_endpoints)),
                "frame_stride": int(state.frame_stride),
                "reward_min": float(out_df["value_current"].min()),
                "reward_max": float(out_df["value_current"].max()),
                "load_parquet_s": round(state.load_parquet_s, 3),
                "decode_frames_s": round(state.decode_frames_s, 3),
                "prepare_prefix_s": round(state.prepare_prefix_s, 3),
                "inference_s": round(inference_s, 3),
                "interpolate_s": round(interp_s, 3),
                "write_parquet_s": round(write_s, 3),
                "episode_total_s": round(time.perf_counter() - state.episode_start_t, 3),
                "part": str(part_path),
                "progress": f"{state.index}/{state.total_episodes}",
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    data_root = args.source_root
    output_root = args.output_root
    meta_dir = output_root / "meta"
    parts_dir = meta_dir / "advantage_robometer-4B.parts"
    final_path = meta_dir / "advantage_robometer-4B.parquet"

    meta_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merged = merge_parts(parts_dir, final_path)
        print(json.dumps({"merged_rows": int(len(merged)), "out": str(final_path)}, indent=2), flush=True)
        return

    episode_paths = select_episode_paths(data_root, args)
    if not episode_paths:
        raise ValueError(f"No episode parquet files found under {data_root / 'data'}")

    dataset_name = output_root.name
    total = len(episode_paths)
    indexed_paths = list(enumerate(episode_paths, start=1))
    indexed_iter = iter(indexed_paths)
    active_states: dict[str, EpisodeState] = {}
    active_order: list[str] = []
    pending_chunks: deque[tuple[str, int, int]] = deque()
    inflight: dict[Any, tuple[str, int, int]] = {}
    target_pending = max(args.parallel_chunks * 2, args.prefetch_episodes)

    def maybe_load_episodes() -> None:
        while len(active_states) < args.prefetch_episodes:
            try:
                index, parquet_path = next(indexed_iter)
            except StopIteration:
                break
            stem = parquet_path.stem
            part_path = parts_dir / f"{stem}.parquet"
            if part_path.exists() and not args.overwrite:
                print(f"[{index}/{total}] skip {stem}", flush=True)
                continue
            state = load_episode_state(
                parquet_path,
                index=index,
                total=total,
                args=args,
                dataset_name=dataset_name,
            )
            active_states[stem] = state
            active_order.append(stem)

    maybe_load_episodes()
    fill_pending_chunks(
        pending_chunks=pending_chunks,
        active_states=active_states,
        active_order=active_order,
        chunk_size=args.chunk_size,
        target_pending=target_pending,
    )

    def run_chunk(item: tuple[str, int, int]) -> tuple[str, int, int, list[float], list[float], dict[str, float]]:
        stem, start, end = item
        state = active_states[stem]
        sample_build_start = time.perf_counter()
        samples = build_prefix_samples_for_range(
            frames=state.frames,
            task=state.task,
            prefix_endpoints=state.prefix_endpoints,
            start=start,
            end=end,
        )
        sample_build_s = time.perf_counter() - sample_build_start
        chunk_start = time.perf_counter()
        result = post_batch(args.eval_server_url, samples, args.timeout_s)
        outputs = result["outputs"]
        progress_pred = outputs.get("outputs_progress", {}).get("progress_pred", [])
        success_probs = outputs.get("outputs_success", {}).get("success_probs", [])
        chunk_rewards = [float(pred[-1]) for pred in progress_pred]
        chunk_success = [float(pred[-1]) for pred in success_probs]
        if len(chunk_success) < len(chunk_rewards):
            chunk_success.extend([0.0] * (len(chunk_rewards) - len(chunk_success)))
        timings = {
            "chunk_total_s": time.perf_counter() - chunk_start,
            "sample_build_s": sample_build_s,
            "payload_s": float(result["payload_s"]),
            "request_s": float(result["request_s"]),
            "response_decode_s": float(result["response_decode_s"]),
        }
        return stem, start, end, chunk_rewards, chunk_success, timings

    with ThreadPoolExecutor(max_workers=args.parallel_chunks) as executor:
        while active_states or pending_chunks or inflight:
            maybe_load_episodes()
            fill_pending_chunks(
                pending_chunks=pending_chunks,
                active_states=active_states,
                active_order=active_order,
                chunk_size=args.chunk_size,
                target_pending=target_pending,
            )

            while pending_chunks and len(inflight) < args.parallel_chunks:
                item = pending_chunks.popleft()
                future = executor.submit(run_chunk, item)
                inflight[future] = item

            if not inflight:
                break

            completed_future = next(as_completed(list(inflight.keys())))
            stem, start, end, chunk_rewards, chunk_success, timings = completed_future.result()
            inflight.pop(completed_future, None)
            state = active_states[stem]
            state.sampled_rewards[start:end] = chunk_rewards
            state.sampled_success_probs[start:end] = chunk_success
            state.chunks_completed += 1
            state.chunk_samples_completed += len(chunk_rewards)
            samples_per_s = len(chunk_rewards) / timings["chunk_total_s"] if timings["chunk_total_s"] > 0 else None
            print(
                json.dumps(
                    {
                        "event": "chunk_done",
                        "episode": stem,
                        "range": [start, end],
                        "num_samples": len(chunk_rewards),
                        "episode_samples_done": int(state.chunk_samples_completed),
                        "episode_total_samples": len(state.prefix_endpoints),
                        "episode_chunks_done": int(state.chunks_completed),
                        "episode_total_chunks": int(state.total_chunks),
                        "chunk_total_s": round(timings["chunk_total_s"], 3),
                        "sample_build_s": round(timings["sample_build_s"], 3),
                        "payload_s": round(timings["payload_s"], 3),
                        "request_s": round(timings["request_s"], 3),
                        "response_decode_s": round(timings["response_decode_s"], 3),
                        "samples_per_s": round(samples_per_s, 3) if samples_per_s is not None else None,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

            if state.chunks_completed >= state.total_chunks:
                part_path = parts_dir / f"{stem}.parquet"
                finalize_episode(state, part_path)
                active_states.pop(stem, None)
                active_order = [item for item in active_order if item != stem]

    merged = merge_parts(parts_dir, final_path)
    print(
        json.dumps(
            {
                "episodes": total,
                "rows": int(len(merged)),
                "parts_dir": str(parts_dir),
                "out": str(final_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
