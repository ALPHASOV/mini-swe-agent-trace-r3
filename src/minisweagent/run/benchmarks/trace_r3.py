#!/usr/bin/env python3

"""Run frozen-validation, baseline-first TRACE-R³ recovery on SWE-bench."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path

import typer
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING, filter_instances
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.trace_r3.controller import TraceR3Controller, TraceR3Result
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_LOCK = threading.Lock()
DEFAULT_CONFIG_SPECS = [
    str(builtin_config_dir / "benchmarks" / "swebench.yaml"),
    str(builtin_config_dir / "benchmarks" / "trace_r3.yaml"),
]


@app.command(
    help=(
        "Freeze an independent pre-patch validation plan, run the unchanged "
        "mini-SWE-agent baseline, then activate recovery only when Gate G0 is not green."
    )
)
def main(
    subset: str = typer.Option("verified", "--subset", help="SWE-bench subset or dataset path"),
    split: str = typer.Option("test", "--split", help="Dataset split"),
    slice_spec: str = typer.Option("", "--slice", help="Slice, for example 0:5"),
    filter_spec: str = typer.Option("", "--filter", help="Instance ID regex"),
    output: str = typer.Option("", "-o", "--output", help="Output directory"),
    workers: int = typer.Option(1, "-w", "--workers", min=1),
    model: str | None = typer.Option(None, "-m", "--model", help="Baseline and recovery model"),
    model_class: str | None = typer.Option(None, "--model-class"),
    redo_existing: bool = typer.Option(False, "--redo-existing"),
    config_spec: list[str] = typer.Option(DEFAULT_CONFIG_SPECS, "-c", "--config"),
    environment_class: str | None = typer.Option(None, "--environment-class"),
) -> None:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    add_file_handler(output_path / "trace_r3.log")
    logger.info(f"TRACE-R³ results will be saved to {output_path}")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec)
    if not redo_existing and (output_path / "preds.json").exists():
        existing = set(json.loads((output_path / "preds.json").read_text()))
        instances = [instance for instance in instances if instance["instance_id"] not in existing]
        logger.info(f"Skipping {len(existing)} existing prediction records")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
        }
    )
    config = recursive_merge(*configs)
    progress = RunBatchProgressManager(len(instances), output_path / f"trace_r3_statuses_{time.time()}.yaml")

    def process(instance: dict) -> None:
        result = TraceR3Controller(
            instance=instance,
            output_dir=output_path,
            config=config,
            progress_manager=progress,
        ).run()
        _update_predictions(output_path / "preds.json", result)

    with Live(progress.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, instance): instance["instance_id"] for instance in instances}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    instance_id = futures[future]
                    logger.error(f"Uncaught TRACE-R³ error for {instance_id}: {error}", exc_info=True)
                    progress.on_uncaught_exception(instance_id, error)


def _update_predictions(path: Path, result: TraceR3Result) -> None:
    with _OUTPUT_LOCK:
        data = json.loads(path.read_text()) if path.exists() else {}
        data[result.instance_id] = {
            "model_name_or_path": result.model_name,
            "instance_id": result.instance_id,
            "model_patch": result.patch,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)


if __name__ == "__main__":
    app()
