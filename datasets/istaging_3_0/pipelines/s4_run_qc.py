"""
Run iSTAGING 3.0 QC pipeline.

Steps:
  1. data_selection / merge_files  — build merged CSV for the QC task (data_desc.json)
  2. data_qc / plots               — run each plot defined in plot_desc.json

Plot types supported: scatter_fit, dist, boxplot, violin, heatmap, pca.

Usage:
  python s4_run_qc.py --in_dir <in_dir> --out_dir <out_dir> --task <task>
  python s4_run_qc.py --in_dir <in_dir> --out_dir <out_dir> --task <task> --plot_desc plot_desc_v2.json  # single file
  python s4_run_qc.py --in_dir <in_dir> --out_dir <out_dir> --task <task> -v
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

_UTILS = Path(__file__).resolve().parent / "utils"
sys.path.insert(0, str(_UTILS))
from logger import setup_logger  # type: ignore[import]

THIS_DIR   = Path(__file__).parent
INPUT_DIR  = None  # set by --in_dir
OUTPUT_DIR = None  # set by --out_dir

PYTHON = sys.executable

TOTAL_STEPS = 2

_log: logging.Logger = logging.getLogger("run_qc")


def step_header(n: int | str, label: str) -> None:
    _log.info(f"\n[STEP {n}/{TOTAL_STEPS}]  {label}")
    _log.debug("=" * 72)


def run(label: str, cmd: list[Any]) -> None:
    _log.info(f"  » {label}")
    full_cmd = [str(c) for c in cmd]
    _log.debug("    CMD: " + " ".join(full_cmd))

    result = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in (result.stdout or "").splitlines():
        _log.debug("    " + line)

    if result.returncode != 0:
        tail = (result.stdout or "").strip().splitlines()[-20:]
        for line in tail:
            _log.error("    " + line)
        _log.error(f"  FAILED: {label}  (exit {result.returncode})")
        sys.exit(1)

    for line in (result.stdout or "").splitlines():
        if "already exists" in line:
            _log.info("    " + line)

    _log.info(f"\n  SUCCESS: {label}")


def _build_plot_cmds(
    plot_cfg: dict[str, Any],
    data_csv: Path,
    output_dir: Path,
) -> list[tuple[str, list[Any]]]:
    """Return (label, cmd) pairs for every call implied by one plot config."""
    ptype  = plot_cfg.get("type", "")
    script = THIS_DIR / f"data_qc/src/plot_{ptype}.py"

    if not script.exists():
        _log.warning(f"  Unknown plot type '{ptype}' — script not found: {script}")
        return []

    hue_vars = plot_cfg.get("hue") or [None]
    col_vars = plot_cfg.get("col") or [None]
    cmds: list[tuple[str, list[Any]]] = []

    def _label(base: str, hue, col) -> str:
        s = base
        if hue: s += f" by {hue}"
        if col: s += f" | col {col}"
        return s

    if ptype == "scatter_fit":
        var_x    = plot_cfg["var_x"]
        fit_type = plot_cfg.get("fit_type")
        for var_y in plot_cfg.get("var_y", []):
            for hue in hue_vars:
                for col in col_vars:
                    cmd: list[Any] = [
                        PYTHON, script,
                        "--data_csv",   data_csv,
                        "--var_x",      var_x,
                        "--var_y",      var_y,
                        "--output_dir", output_dir,
                    ]
                    if fit_type: cmd += ["--fit_type", fit_type]
                    if hue:      cmd += ["--hue", hue]
                    if col:      cmd += ["--col", col]
                    cmds.append((_label(f"scatter_fit ({var_y} vs {var_x})", hue, col), cmd))

    elif ptype == "dist":
        for var_y in plot_cfg.get("var_y", []):
            for hue in hue_vars:
                for col in col_vars:
                    cmd = [
                        PYTHON, script,
                        "--data_csv",   data_csv,
                        "--var_y",      var_y,
                        "--output_dir", output_dir,
                    ]
                    if hue: cmd += ["--hue", hue]
                    if col: cmd += ["--col", col]
                    if plot_cfg.get("title"): cmd += ["--title", plot_cfg["title"]]
                    cmds.append((_label(f"dist ({var_y})", hue, col), cmd))

    elif ptype in ("boxplot", "violin"):
        var_x = plot_cfg["var_x"]
        for var_y in plot_cfg.get("var_y", []):
            for col in col_vars:
                cmd = [
                    PYTHON, script,
                    "--data_csv",   data_csv,
                    "--var_x",      var_x,
                    "--var_y",      var_y,
                    "--output_dir", output_dir,
                ]
                if col: cmd += ["--col", col]
                if plot_cfg.get("title"): cmd += ["--title", plot_cfg["title"]]
                cmds.append((_label(f"{ptype} ({var_y} by {var_x})", None, col), cmd))

    elif ptype == "heatmap":
        name = plot_cfg.get("name", "heatmap")
        for col in col_vars:
            cmd = [
                PYTHON, script,
                "--data_csv",   data_csv,
                "--name",       name,
                "--output_dir", output_dir,
                "--var_y",
            ] + list(plot_cfg.get("var_y", []))
            if col: cmd += ["--col", col]
            if plot_cfg.get("title"): cmd += ["--title", plot_cfg["title"]]
            cmds.append((_label(f"heatmap ({name})", None, col), cmd))

    elif ptype == "pca":
        name = plot_cfg.get("name", "pca")
        for hue in hue_vars:
            for col in col_vars:
                cmd = [
                    PYTHON, script,
                    "--data_csv",   data_csv,
                    "--name",       name,
                    "--output_dir", output_dir,
                    "--var_y",
                ] + list(plot_cfg.get("var_y", []))
                if hue: cmd += ["--hue", hue]
                if col: cmd += ["--col", col]
                if plot_cfg.get("title"): cmd += ["--title", plot_cfg["title"]]
                cmds.append((_label(f"pca ({name})", hue, col), cmd))

    return cmds


def main():
    parser = argparse.ArgumentParser(description="Run iSTAGING 3.0 QC pipeline.")
    parser.add_argument("--in_dir",  required=True,
                        help="anonymized input directory (contains tasks/, dictionaries/)")
    parser.add_argument("--out_dir", required=True,
                        help="output directory root")
    parser.add_argument(
        "--task", required=True, metavar="TASK",
        help="QC task folder name under in_dir/tasks/",
    )
    parser.add_argument(
        "--plot_desc", default=None, metavar="FILE",
        help="plot description filename inside the task folder "
             "(default: all plot_desc*.json files in the task folder)",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug output on the console")
    parser.add_argument("--log_dir", default=None,
                        help="directory for log file (default: out_dir/logs)")
    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_DIR
    INPUT_DIR  = Path(args.in_dir)
    OUTPUT_DIR = Path(args.out_dir)

    setup_logger("run_qc", verbose=args.verbose,
                 log_dir=Path(args.log_dir) if args.log_dir else OUTPUT_DIR / "logs")
    _log.debug("Command: " + " ".join(sys.argv))

    task = args.task
    _log.info(f"\n=== iSTAGING 3.0 QC pipeline  task={task} ===")

    # ── Step 1: build merged feature CSV ─────────────────────────────────────
    step_header(1, f"data_selection / merge_files  {task}")
    run(f"merge_files ({task})", [
        PYTHON, THIS_DIR / "data_utils/data_selection/src/merge_files.py",
        "--task_dir",    INPUT_DIR / "tasks" / task,
        "--desc_file",   "data_desc.json",
        "--var_groups",  INPUT_DIR / "dictionaries/dict_var_groups.json",
        "--data_files",  INPUT_DIR / "dictionaries/dict_data_files.json",
        "--root",        INPUT_DIR,
        "--output_root", OUTPUT_DIR,
    ])

    # ── Step 2: plots — driven by plot_desc*.json ────────────────────────────
    task_dir = INPUT_DIR / "tasks" / task
    if args.plot_desc:
        plot_desc_files = [task_dir / args.plot_desc]
    else:
        plot_desc_files = sorted(task_dir.glob("plot_desc*.json"))

    step_header(2, f"data_qc / plots  {task}  ({len(plot_desc_files)} desc file(s))")

    if not plot_desc_files:
        _log.warning(f"  No plot_desc*.json files found in {task_dir} — skipping")

    for plot_desc_path in plot_desc_files:
        if not plot_desc_path.exists():
            _log.warning(f"  Plot desc not found: {plot_desc_path} — skipping")
            continue

        plot_desc = json.loads(plot_desc_path.read_text(encoding="utf-8"))
        data_csv   = OUTPUT_DIR / plot_desc["data"]["path"] / plot_desc["data"]["file"]
        output_dir = OUTPUT_DIR / plot_desc["output"]["path"]
        plots      = plot_desc.get("plots", [])

        _log.info(f"  {plot_desc_path.name}  ({len(plots)} plot configs)")

        for plot_cfg in plots:
            ptype = plot_cfg.get("type", "?")
            name  = plot_cfg.get("name", "")
            _log.debug(f"    plot type={ptype} name={name}")

            for label, cmd in _build_plot_cmds(plot_cfg, data_csv, output_dir):
                run(label, cmd)

    _log.info("\nQC pipeline completed.")


if __name__ == "__main__":
    main()
