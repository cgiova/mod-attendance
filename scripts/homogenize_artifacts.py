#!/usr/bin/env python3
"""Normalize legacy artifact run directories.

The current notebook stores runs as:

    artifacts/enhanced/runs/<strategy>/<study_tag>/seed_<seed>/<setting>__e<epochs>

Older runs used a few flatter layouts. This script detects those legacy run
directories, moves them into the current layout, rewrites local metadata, and
creates study-level combined/aggregate CSVs. It dry-runs by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isic24.run_ids import make_exp_tag, make_run_dir, make_run_id, make_setting_folder  # noqa: E402


KNOWN_IMPLS = (
    "vanilla",
    "fixed_scaling",
    "learnable_scaling",
    "loss_regularization",
    "controlled_fusion",
    "adaptive_gating",
)

SPLIT_SEED_IMPLS = {"controlled_fusion", "adaptive_gating"}

DEFAULT_SETTING_TAG = {
    "vanilla": "baseline",
    "controlled_fusion": "base",
    "adaptive_gating": "base",
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "controlled_fusion": {"gamma_img": 1.0, "gamma_tab": 1.0, "tag": "base"},
    "adaptive_gating": {"entropy_weight": 0.0001, "tag": "base"},
}

AGGREGATES = {
    "fixed_scaling": [("alpha_aggregated.csv", ["alpha"], "best_val_auc", ("mean", "std", "count"))],
    "learnable_scaling": [
        ("init_alpha_aggregated.csv", ["init_alpha"], "best_val_auc", ("mean", "std", "count")),
        ("alpha_final_aggregated.csv", ["init_alpha"], "alpha_final", ("mean", "std")),
    ],
    "loss_regularization": [
        ("lambda_reg_aggregated.csv", ["lambda_reg"], "best_val_auc", ("mean", "std", "count"))
    ],
    "controlled_fusion": [
        (
            "controlled_fusion_aggregated.csv",
            ["setting_tag", "gamma_img", "gamma_tab"],
            "best_val_auc",
            ("mean", "std", "count"),
        )
    ],
    "adaptive_gating": [
        ("entropy_weight_aggregated.csv", ["entropy_weight"], "best_val_auc", ("mean", "std", "count"))
    ],
}

REQUIRED_SUMMARY_FIELDS = (
    "impl_name",
    "exp_tag",
    "setting_tag",
    "run_id",
    "seed",
    "best_epoch",
    "best_val_auc",
    "train_seconds",
    "xai_done",
    "run_dir",
    "model_name",
)


@dataclass
class Candidate:
    impl: str
    old_study_tag: str
    old_run_dir: Path
    run_folder: str
    layout_kind: str
    seed: int
    epochs: int
    setting_tag: str
    config: dict[str, Any]
    summary_row: dict[str, str] | None
    summary_fields: list[str]


@dataclass
class MovePlan:
    candidate: Candidate
    study_tag: str
    new_run_dir: Path
    new_exp_tag: str
    new_run_id: str
    new_model_name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def old_run_dir(self) -> Path:
        return self.candidate.old_run_dir

    @property
    def impl(self) -> str:
        return self.candidate.impl

    @property
    def seed(self) -> int:
        return self.candidate.seed

    @property
    def setting_tag(self) -> str:
        return self.candidate.setting_tag


@dataclass
class StudyPlan:
    impl: str
    study_tag: str
    layout_kind: str
    moves: list[MovePlan]

    @property
    def study_dir(self) -> Path:
        return self.moves[0].new_run_dir.parents[1]


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "null"}


def value_from(*values: Any, default: Any = None) -> Any:
    for value in values:
        if not is_blank(value):
            return value
    return default


def parse_int(value: Any, *, label: str) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not parse {label}: {value!r}") from exc


def parse_float(value: Any) -> float | None:
    if is_blank(value):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def csv_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_single_summary(path: Path) -> tuple[list[str], dict[str, str] | None]:
    if not path.exists():
        return [], None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return list(reader.fieldnames or []), None
    return list(reader.fieldnames or []), rows[0]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: csv_string(row.get(field_name, "")) for field_name in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def backup_file(path: Path, stamp: str, backups: list[dict[str, str]]) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.pre_homogenize_{stamp}.bak")
    suffix = 1
    while backup.exists():
        backup = path.with_name(f"{path.name}.pre_homogenize_{stamp}.{suffix}.bak")
        suffix += 1
    shutil.copy2(path, backup)
    backups.append({"path": str(path), "backup": str(backup)})


def ensure_fields(fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
    ordered = list(fields)
    for required in REQUIRED_SUMMARY_FIELDS:
        if required not in ordered:
            ordered.append(required)
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def infer_epochs(run_folder: str, config: dict[str, Any], row: dict[str, str] | None) -> int:
    value = value_from(config.get("epochs"), row.get("epochs") if row else None)
    if not is_blank(value):
        return parse_int(value, label="epochs")
    match = re.search(r"__e(\d+)$", run_folder)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer epochs from {run_folder!r}")


def infer_setting_tag(impl: str, run_folder: str, config: dict[str, Any], row: dict[str, str] | None) -> str:
    setting = value_from(
        row.get("setting_tag") if row else None,
        config.get("setting_tag"),
        config.get("tag"),
        default=None,
    )
    if is_blank(setting) and not run_folder.startswith("run__"):
        setting = re.sub(r"__e\d+$", "", run_folder)
    if is_blank(setting):
        setting = DEFAULT_SETTING_TAG.get(impl, "run")
    return str(setting)


def infer_seed(config: dict[str, Any], row: dict[str, str] | None) -> int:
    value = value_from(row.get("seed") if row else None, config.get("seed"))
    return parse_int(value, label="seed")


def infer_params(impl: str, setting_tag: str, config: dict[str, Any], row: dict[str, str] | None) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS.get(impl, {}))
    params["tag"] = value_from(row.get("tag") if row else None, config.get("tag"), params.get("tag"), setting_tag)
    params["tag"] = setting_tag if is_blank(params.get("tag")) else params["tag"]

    if impl == "controlled_fusion":
        params["gamma_img"] = parse_float(value_from(row.get("gamma_img") if row else None, config.get("gamma_img"), 1.0))
        params["gamma_tab"] = parse_float(value_from(row.get("gamma_tab") if row else None, config.get("gamma_tab"), 1.0))
    elif impl == "adaptive_gating":
        params["entropy_weight"] = parse_float(
            value_from(row.get("entropy_weight") if row else None, config.get("entropy_weight"), 0.0001)
        )
    return params


def candidate_sort_key(candidate: Candidate) -> tuple[str, str, int, str]:
    return (candidate.impl, candidate.old_study_tag, candidate.seed, candidate.run_folder)


def discover_candidates(runs_root: Path, impls: set[str]) -> tuple[list[Candidate], list[str]]:
    run_dirs: set[Path] = set()
    warnings: list[str] = []

    for impl in sorted(impls):
        impl_root = runs_root / impl
        if not impl_root.exists():
            warnings.append(f"missing strategy folder: {impl_root}")
            continue
        for config_path in impl_root.rglob("config.json"):
            run_dirs.add(config_path.parent)
        for summary_path in impl_root.rglob("single_run_summary.csv"):
            run_dirs.add(summary_path.parent)

    candidates: list[Candidate] = []
    for run_dir in sorted(run_dirs):
        try:
            rel = run_dir.relative_to(runs_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        impl, old_study_tag, run_folder = parts
        if impl not in impls:
            continue

        config = read_json(run_dir / "config.json")
        summary_fields, summary_row = read_single_summary(run_dir / "single_run_summary.csv")

        try:
            seed = infer_seed(config, summary_row)
            epochs = infer_epochs(run_folder, config, summary_row)
            setting_tag = infer_setting_tag(impl, run_folder, config, summary_row)
        except ValueError as exc:
            warnings.append(f"skipping {run_dir}: {exc}")
            continue

        layout_kind = "split_seed" if impl in SPLIT_SEED_IMPLS and run_folder.startswith("run__") else "flat_study"
        candidates.append(
            Candidate(
                impl=impl,
                old_study_tag=old_study_tag,
                old_run_dir=run_dir,
                run_folder=run_folder,
                layout_kind=layout_kind,
                seed=seed,
                epochs=epochs,
                setting_tag=setting_tag,
                config=config,
                summary_row=summary_row,
                summary_fields=summary_fields,
            )
        )

    return sorted(candidates, key=candidate_sort_key), warnings


def group_split_candidates(candidates: list[Candidate]) -> list[list[Candidate]]:
    groups: list[list[Candidate]] = []
    current: list[Candidate] = []
    seen_seeds: set[int] = set()

    for candidate in sorted(candidates, key=lambda c: c.old_study_tag):
        if current and candidate.seed in seen_seeds:
            groups.append(current)
            current = []
            seen_seeds = set()
        current.append(candidate)
        seen_seeds.add(candidate.seed)
    if current:
        groups.append(current)
    return groups


def build_study_plans(candidates: list[Candidate], runs_root: Path) -> list[StudyPlan]:
    plans: list[StudyPlan] = []
    split_by_impl: dict[str, list[Candidate]] = {}
    flat_by_study: dict[tuple[str, str], list[Candidate]] = {}

    for candidate in candidates:
        if candidate.layout_kind == "split_seed":
            split_by_impl.setdefault(candidate.impl, []).append(candidate)
        else:
            flat_by_study.setdefault((candidate.impl, candidate.old_study_tag), []).append(candidate)

    for impl, impl_candidates in sorted(split_by_impl.items()):
        for group in group_split_candidates(impl_candidates):
            study_tag = group[0].old_study_tag
            plans.append(make_study_plan(impl, study_tag, "split_seed", group, runs_root))

    for (impl, study_tag), group in sorted(flat_by_study.items()):
        plans.append(make_study_plan(impl, study_tag, "flat_study", group, runs_root))

    return sorted(plans, key=lambda plan: (plan.impl, plan.study_tag, plan.layout_kind))


def make_study_plan(
    impl: str,
    study_tag: str,
    layout_kind: str,
    group: list[Candidate],
    runs_root: Path,
) -> StudyPlan:
    moves: list[MovePlan] = []
    for candidate in sorted(group, key=lambda c: (c.seed, c.setting_tag, c.old_study_tag)):
        params = infer_params(candidate.impl, candidate.setting_tag, candidate.config, candidate.summary_row)
        # Le formule di run_id/run_dir stanno in `isic24.run_ids`, non qui: erano replicate
        # in tre posti e le tre copie non concordavano.  Questa, in particolare, non gestiva
        # `setting_tag=None` e avrebbe scritto "None__e20" dove il training scrive "run__e20".
        new_exp_tag = make_exp_tag(study_tag, candidate.seed)
        new_setting_folder = make_setting_folder(candidate.setting_tag, candidate.epochs)
        new_run_id = make_run_id(new_exp_tag, new_setting_folder)
        new_model_name = f"{candidate.impl}__{candidate.setting_tag}" if candidate.setting_tag else candidate.impl
        new_run_dir = Path(make_run_dir(runs_root, candidate.impl, new_exp_tag, new_setting_folder))
        moves.append(
            MovePlan(
                candidate=candidate,
                study_tag=study_tag,
                new_run_dir=new_run_dir,
                new_exp_tag=new_exp_tag,
                new_run_id=new_run_id,
                new_model_name=new_model_name,
                params=params,
            )
        )
    return StudyPlan(impl=impl, study_tag=study_tag, layout_kind=layout_kind, moves=moves)


def updated_summary_row(move: MovePlan) -> dict[str, Any] | None:
    old_row = move.candidate.summary_row
    if old_row is None:
        return None
    row: dict[str, Any] = dict(old_row)
    row.update(
        {
            "impl_name": move.impl,
            "exp_tag": move.new_exp_tag,
            "setting_tag": move.setting_tag,
            "run_id": move.new_run_id,
            "seed": move.seed,
            "run_dir": str(move.new_run_dir.resolve()),
            "model_name": move.new_model_name,
        }
    )
    for key, value in move.params.items():
        if key == "tag" and is_blank(value):
            value = move.setting_tag
        row[key] = value
    return row


def updated_config(move: MovePlan) -> dict[str, Any]:
    config = dict(move.candidate.config)
    config.update(
        {
            "impl_name": move.impl,
            "exp_tag": move.new_exp_tag,
            "setting_tag": move.setting_tag,
            "run_id": move.new_run_id,
            "model_name": move.new_model_name,
            "epochs": move.candidate.epochs,
            "seed": move.seed,
        }
    )
    for key, value in move.params.items():
        if key == "tag" and is_blank(value):
            value = move.setting_tag
        config[key] = value
    return config


def check_collisions(plans: list[StudyPlan]) -> list[str]:
    errors: list[str] = []
    destinations: dict[Path, Path] = {}
    for plan in plans:
        for move in plan.moves:
            dest = move.new_run_dir
            if dest in destinations:
                errors.append(f"duplicate destination: {dest}")
            destinations[dest] = move.old_run_dir
            if dest.exists() and dest.resolve() != move.old_run_dir.resolve():
                errors.append(f"destination already exists: {dest}")
    return errors


def move_root_seed_artifacts(
    plan: StudyPlan,
    execute: bool,
    stamp: str,
    backups: list[dict[str, str]],
    manifest_moves: list[dict[str, str]],
    warnings: list[str],
) -> None:
    if plan.layout_kind != "flat_study":
        return
    seeds = sorted({move.seed for move in plan.moves})
    if len(seeds) != 1:
        warnings.append(f"leaving root summary/plots in {plan.study_dir}: multiple seeds in flat legacy study")
        return

    seed_dir = plan.study_dir / f"seed_{seeds[0]}"
    root_summary = plan.study_dir / "summary.csv"
    seed_summary = seed_dir / "summary.csv"
    if root_summary.exists():
        manifest_moves.append({"from": str(root_summary), "to": str(seed_summary)})
        if execute:
            seed_summary.parent.mkdir(parents=True, exist_ok=True)
            if seed_summary.exists():
                warnings.append(f"not moving {root_summary}: {seed_summary} already exists")
            else:
                shutil.move(str(root_summary), str(seed_summary))
                backup_file(seed_summary, stamp, backups)

    root_plots = plan.study_dir / "plots"
    seed_plots = seed_dir / "plots"
    if root_plots.exists():
        manifest_moves.append({"from": str(root_plots), "to": str(seed_plots)})
        if execute:
            seed_plots.parent.mkdir(parents=True, exist_ok=True)
            if seed_plots.exists():
                warnings.append(f"not moving {root_plots}: {seed_plots} already exists")
            else:
                shutil.move(str(root_plots), str(seed_plots))


def rename_checkpoint(move: MovePlan, warnings: list[str], manifest_moves: list[dict[str, str]]) -> None:
    old_model_name = value_from(move.candidate.config.get("model_name"), move.impl)
    old_ckpt = move.new_run_dir / "checkpoints" / f"best_{old_model_name}.keras"
    new_ckpt = move.new_run_dir / "checkpoints" / f"best_{move.new_model_name}.keras"
    if old_ckpt == new_ckpt:
        return
    if old_ckpt.exists() and not new_ckpt.exists():
        old_ckpt.rename(new_ckpt)
        manifest_moves.append({"from": str(old_ckpt), "to": str(new_ckpt)})
    elif old_ckpt.exists() and new_ckpt.exists():
        warnings.append(f"checkpoint rename skipped because target exists: {new_ckpt}")
    elif not new_ckpt.exists():
        warnings.append(f"checkpoint not found for {move.new_run_dir}: expected {old_ckpt.name}")


def write_run_metadata(move: MovePlan, stamp: str, backups: list[dict[str, str]]) -> None:
    config_path = move.new_run_dir / "config.json"
    backup_file(config_path, stamp, backups)
    write_json(config_path, updated_config(move))

    row = updated_summary_row(move)
    if row is None:
        return
    summary_path = move.new_run_dir / "single_run_summary.csv"
    backup_file(summary_path, stamp, backups)
    fields = ensure_fields(move.candidate.summary_fields, [row])
    write_csv(summary_path, fields, [row])


def rows_for_plan(plan: StudyPlan) -> list[dict[str, Any]]:
    rows = []
    for move in plan.moves:
        row = updated_summary_row(move)
        if row is not None:
            rows.append(row)
    return rows


def fields_for_rows(moves: list[MovePlan], rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for move in moves:
        for field_name in move.candidate.summary_fields:
            if field_name not in fields:
                fields.append(field_name)
    return ensure_fields(fields, rows)


def write_seed_and_combined_summaries(plan: StudyPlan) -> None:
    rows = rows_for_plan(plan)
    if not rows:
        return
    fields = fields_for_rows(plan.moves, rows)

    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(parse_int(row["seed"], label="seed"), []).append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        write_csv(plan.study_dir / f"seed_{seed}" / "summary.csv", fields, seed_rows)
    write_csv(plan.study_dir / "combined_summary.csv", fields, rows)
    write_strategy_aggregates(plan.impl, plan.study_dir, rows)


def aggregate_rows(
    rows: list[dict[str, Any]],
    group_cols: list[str],
    metric_col: str,
    stats: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        value = parse_float(row.get(metric_col))
        if value is None:
            continue
        key = tuple(csv_string(row.get(col, "")) for col in group_cols)
        grouped.setdefault(key, []).append(value)

    out_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        out: dict[str, Any] = {col: key[idx] for idx, col in enumerate(group_cols)}
        if "mean" in stats:
            out["mean"] = statistics.mean(values)
        if "std" in stats:
            out["std"] = statistics.stdev(values) if len(values) > 1 else ""
        if "count" in stats:
            out["count"] = len(values)
        out_rows.append(out)
    return out_rows


def write_strategy_aggregates(impl: str, study_dir: Path, rows: list[dict[str, Any]]) -> None:
    for filename, group_cols, metric_col, stats in AGGREGATES.get(impl, []):
        if any(col not in rows[0] for col in group_cols) or metric_col not in rows[0]:
            continue
        agg_rows = aggregate_rows(rows, group_cols, metric_col, stats)
        if not agg_rows:
            continue
        fields = list(group_cols) + list(stats)
        write_csv(study_dir / filename, fields, agg_rows)


def update_faithfulness_csv(
    runs_root: Path,
    plans: list[StudyPlan],
    execute: bool,
    stamp: str,
    backups: list[dict[str, str]],
    warnings: list[str],
) -> int:
    faith_csv = runs_root / "_faithfulness" / "combined_faithfulness.csv"
    if not faith_csv.exists():
        return 0

    path_to_move: dict[str, MovePlan] = {}
    for plan in plans:
        for move in plan.moves:
            path_to_move[str(move.old_run_dir.resolve())] = move
            if move.candidate.summary_row and not is_blank(move.candidate.summary_row.get("run_dir")):
                path_to_move[str(Path(move.candidate.summary_row["run_dir"]).resolve())] = move

    fields, rows = read_csv(faith_csv)
    changed = 0
    for row in rows:
        run_dir = row.get("run_dir")
        if is_blank(run_dir):
            continue
        move = path_to_move.get(str(Path(run_dir).resolve()))
        if move is None:
            continue
        row["run_dir"] = str(move.new_run_dir.resolve())
        row["setting_tag"] = move.setting_tag
        row["impl_name"] = move.impl
        for key, value in move.params.items():
            if key in fields:
                row[key] = csv_string(value)
        changed += 1

    if changed and execute:
        backup_file(faith_csv, stamp, backups)
        write_csv(faith_csv, fields, rows)
    elif changed:
        warnings.append(f"would update {changed} rows in {faith_csv}")
    return changed


def execute_plans(plans: list[StudyPlan], runs_root: Path, stamp: str) -> dict[str, Any]:
    backups: list[dict[str, str]] = []
    manifest_moves: list[dict[str, str]] = []
    warnings: list[str] = []

    for plan in plans:
        move_root_seed_artifacts(plan, True, stamp, backups, manifest_moves, warnings)

        for move in plan.moves:
            move.new_run_dir.parent.mkdir(parents=True, exist_ok=True)
            manifest_moves.append({"from": str(move.old_run_dir), "to": str(move.new_run_dir)})
            shutil.move(str(move.old_run_dir), str(move.new_run_dir))
            rename_checkpoint(move, warnings, manifest_moves)
            write_run_metadata(move, stamp, backups)

        write_seed_and_combined_summaries(plan)

        for move in plan.moves:
            old_parent = move.old_run_dir.parent
            if old_parent.exists() and old_parent != plan.study_dir:
                try:
                    old_parent.rmdir()
                except OSError:
                    pass

    faithfulness_updates = update_faithfulness_csv(runs_root, plans, True, stamp, backups, warnings)

    manifest = {
        "created_at": stamp,
        "runs_root": str(runs_root),
        "studies": [
            {
                "impl": plan.impl,
                "study_tag": plan.study_tag,
                "layout_kind": plan.layout_kind,
                "runs": [
                    {
                        "old_run_dir": str(move.old_run_dir),
                        "new_run_dir": str(move.new_run_dir),
                        "seed": move.seed,
                        "setting_tag": move.setting_tag,
                        "old_exp_tag": value_from(
                            move.candidate.summary_row.get("exp_tag") if move.candidate.summary_row else None,
                            move.candidate.config.get("exp_tag"),
                        ),
                        "new_exp_tag": move.new_exp_tag,
                        "old_run_id": value_from(
                            move.candidate.summary_row.get("run_id") if move.candidate.summary_row else None,
                            move.candidate.config.get("run_id"),
                        ),
                        "new_run_id": move.new_run_id,
                        "old_model_name": value_from(move.candidate.config.get("model_name"), move.impl),
                        "new_model_name": move.new_model_name,
                        "has_summary": move.candidate.summary_row is not None,
                    }
                    for move in plan.moves
                ],
            }
            for plan in plans
        ],
        "file_moves": manifest_moves,
        "backups": backups,
        "faithfulness_rows_updated": faithfulness_updates,
        "warnings": warnings,
    }

    manifest_path = runs_root / f"homogenize_manifest_{stamp}.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def print_plan(plans: list[StudyPlan], warnings: list[str], faithfulness_updates: int) -> None:
    n_runs = sum(len(plan.moves) for plan in plans)
    n_summary = sum(1 for plan in plans for move in plan.moves if move.candidate.summary_row is not None)
    print(f"Planned studies: {len(plans)}")
    print(f"Planned run directory moves: {n_runs} ({n_summary} with single_run_summary.csv)")
    print(f"Faithfulness rows to update: {faithfulness_updates}")
    print()
    for plan in plans:
        seeds = ",".join(str(seed) for seed in sorted({move.seed for move in plan.moves}))
        complete = "" if all(move.candidate.summary_row is not None for move in plan.moves) else " [partial]"
        print(
            f"- {plan.impl}/{plan.study_tag} ({plan.layout_kind}, seeds={seeds}, runs={len(plan.moves)}){complete}"
        )
        for move in plan.moves:
            rel_old = move.old_run_dir
            rel_new = move.new_run_dir
            print(f"  {rel_old} -> {rel_new}")
    if warnings:
        print()
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="artifacts/enhanced/runs", type=Path)
    parser.add_argument(
        "--impl",
        action="append",
        choices=KNOWN_IMPLS,
        help="Strategy to migrate. Repeatable. Defaults to all known strategies.",
    )
    parser.add_argument("--execute", action="store_true", help="Apply the migration. Omit for dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    impls = set(args.impl or KNOWN_IMPLS)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    candidates, warnings = discover_candidates(runs_root, impls)
    plans = build_study_plans(candidates, runs_root)
    collisions = check_collisions(plans)
    if collisions:
        print("Refusing to continue because destination collisions were found:")
        for collision in collisions:
            print(f"- {collision}")
        return 2

    faithfulness_updates = update_faithfulness_csv(runs_root, plans, False, stamp, [], warnings)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"[{mode}] artifact homogenization")
    print(f"Runs root: {runs_root}")
    print()
    print_plan(plans, warnings, faithfulness_updates)

    if not args.execute:
        print()
        print("No files changed. Re-run with --execute to apply this plan.")
        return 0

    manifest = execute_plans(plans, runs_root, stamp)
    print()
    print(f"Applied migration. Manifest: {manifest['manifest_path']}")
    if manifest["warnings"]:
        print("Warnings were recorded in the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
