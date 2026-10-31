"""Reproducible greedy self-play with per-turn and per-episode measurements."""

import argparse
import csv
import ctypes
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import gymnasium
import numpy as np

from .environment import ScrabbleEnv, greedy_action


def process_memory_bytes() -> tuple[int, int]:
    """Current and process-lifetime peak working set on Windows (RSS elsewhere)."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    if sys.platform.startswith("linux"):
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        import os
        return pages * os.sysconf("SC_PAGE_SIZE"), peak_bytes
    raise NotImplementedError("Current RSS measurement supports Windows and Linux")


def run_benchmark(
    *, episodes: int = 3, seed: int = 42, max_turns: int = 200,
    word_list: str | Path | None = None, output: str | Path = "scrabble\\results",
    render: bool = False,
) -> dict:
    if episodes < 1 or max_turns < 1:
        raise ValueError("episodes and max_turns must be positive")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    turns = []
    episode_results = []
    tracemalloc.start()
    start = time.perf_counter()
    env = None
    try:
        env = ScrabbleEnv(word_list=word_list, max_turns=max_turns, render_mode="ansi")
        setup_seconds = time.perf_counter() - start
        for episode in range(episodes):
            tracemalloc.reset_peak()
            episode_start = time.perf_counter()
            observation, _ = env.reset(seed=seed + episode)
            done = False
            counts = []
            while not done:
                turn_start = time.perf_counter()
                acting_player = int(observation["player"])
                generation_start = time.perf_counter()
                moves = env.legal_moves()
                generation_seconds = time.perf_counter() - generation_start
                placements = sum(item.move.kind == "place" for item in moves)
                exchanges = sum(item.move.kind == "exchange" for item in moves)
                counts.append(placements)
                selection_start = time.perf_counter()
                action = greedy_action(env)
                selection_seconds = time.perf_counter() - selection_start
                step_start = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                step_seconds = time.perf_counter() - step_start
                turn_seconds = time.perf_counter() - turn_start
                traced_current, traced_peak = tracemalloc.get_traced_memory()
                rss, process_peak = process_memory_bytes()
                move = info["move"]
                turns.append({
                    "episode": episode, "seed": seed + episode,
                    "turn": env.turn, "player": acting_player,
                    "legal_placements": placements, "legal_exchanges": exchanges,
                    "legal_actions_including_pass": len(moves),
                    "generation_seconds": generation_seconds,
                    "selection_seconds": selection_seconds,
                    "step_seconds": step_seconds, "turn_seconds": turn_seconds,
                    "kind": move.kind, "word": move.word, "row": move.row,
                    "col": move.col, "direction": move.direction,
                    "blank_positions": ",".join(map(str, move.blanks)),
                    "exchange": move.exchange, "move_score": info["move_score"],
                    "reward": reward, "score_0": int(env.scores[0]),
                    "score_1": int(env.scores[1]), "bag_size": len(env.bag),
                    "python_current_bytes": traced_current,
                    "python_peak_bytes": traced_peak,
                    "process_rss_bytes": rss, "process_lifetime_peak_bytes": process_peak,
                })
                done = terminated or truncated
                if render:
                    print(env.render())
            episode_seconds = time.perf_counter() - episode_start
            _, python_peak = tracemalloc.get_traced_memory()
            rss, process_peak = process_memory_bytes()
            episode_results.append({
                "episode": episode, "seed": seed + episode, "turns": env.turn,
                "scores": env.scores.tolist(), "winner": info["winner"],
                "terminated": terminated, "truncated": truncated,
                "end_reason": info["end_reason"],
                "legal_placements_mean": statistics.mean(counts),
                "legal_placements_min": min(counts),
                "legal_placements_max": max(counts),
                "runtime_seconds": episode_seconds,
                "python_peak_bytes": python_peak,
                "process_rss_bytes": rss,
                "process_lifetime_peak_bytes": process_peak,
            })
    finally:
        if env is not None:
            env.close()
        tracemalloc.stop()
    report = {
        "configuration": {
            "episodes": episodes, "initial_seed": seed, "max_turns": max_turns,
            "dictionary": str(Path(word_list).resolve()) if word_list else "bundled teaching",
            "dictionary_size": len(env.words), "policy": "greedy vs greedy",
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(),
            "gymnasium": gymnasium.__version__, "numpy": np.__version__,
            "memory_note": (
                "Python peaks use tracemalloc; tracing is enabled during all timings. "
                "Episode peaks include live allocations from setup and retained earlier "
                "CSV rows. RSS includes interpreter/native allocations; process peak "
                "is a process-lifetime high-water mark, not a per-episode delta."
            ),
            "timing_note": (
                "Episode runtime includes reset, bookkeeping, and memory sampling, "
                "but excludes setup and final file writes. Turn runtime excludes "
                "memory sampling, CSV row construction, and optional rendering."
            ),
        },
        "setup_seconds": setup_seconds,
        "summary": {
            "episode_length_mean": statistics.mean(e["turns"] for e in episode_results),
            "episode_runtime_seconds_mean": statistics.mean(
                e["runtime_seconds"] for e in episode_results
            ),
            "legal_placements_per_turn_mean": statistics.mean(
                row["legal_placements"] for row in turns
            ),
            "generation_seconds_mean": statistics.mean(
                row["generation_seconds"] for row in turns
            ),
            "python_peak_bytes": max(e["python_peak_bytes"] for e in episode_results),
            "process_lifetime_peak_bytes": max(
                e["process_lifetime_peak_bytes"] for e in episode_results
            ),
        },
        "episodes": episode_results,
    }
    with (destination / "turns.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(turns[0]))
        writer.writeheader()
        writer.writerows(turns)
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--word-list", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_turns < 1:
        parser.error("--episodes and --max-turns must be positive")
    report = run_benchmark(
        episodes=args.episodes, seed=args.seed, max_turns=args.max_turns,
        word_list=args.word_list, output=args.output, render=args.render,
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Measurements saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
