"""Frozen-gamma explanations, isolated from the canonical training runs."""

from __future__ import annotations

import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths, xai_driver as xd, xai_faithfulness as xf, xai_subset as xs


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete(directory, names):
    ledger_path = directory / "completed.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for name in names:
        if name not in ledger:
            continue
        for relative, digest in ledger[name].items():
            path = directory / relative
            if not path.is_file() or _sha256(path) != digest:
                raise RuntimeError(f"Changed or missing completed {name} output: {path}")
    return all(name in ledger for name in names)


def _stage(directory, name, patterns, action):
    if _complete(directory, (name,)):
        return
    print(f"[CF XAI] {directory.parent.name}/{directory.name}: {name}", flush=True)
    action()
    files = []
    for pattern in patterns:
        found = sorted(directory.glob(pattern))
        if not found:
            raise RuntimeError(f"Missing {name} output: {directory / pattern}")
        files.extend(path for path in found if path.is_file())
    ledger_path = directory / "completed.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ledger[name] = {str(path.relative_to(directory)): _sha256(path) for path in files}
    _write_json(ledger_path, ledger)


def _check_array_sample(path, indices):
    with np.load(path) as data:
        expected = {"val_index": indices, **xs.stamp_row_ids(indices)}
        for key, value in expected.items():
            if key not in data or not np.array_equal(data[key], value):
                raise RuntimeError(f"Sample identity mismatch in {path}: {key}")


def _callbacks(model, row, directory, engines, indices):
    callbacks = xd.make_xai_callbacks(
        str(directory), row["base_model_name"], engines=engines)
    status = xd.run_xai_on_best_epoch(
        model, callbacks, int(row["best_epoch"]), float(row["val_auc"]),
        run_dir=str(directory))
    if not status or any(result != "ok" for result in status.values()):
        raise RuntimeError(f"Incomplete gamma XAI at {directory}: {status}")
    if "shap" in engines:
        arrays = directory / "xai" / "shap_tab" / f"epoch_{int(row['best_epoch']):03d}"
        _check_array_sample(arrays / f"shap_arrays_epoch_{int(row['best_epoch']):03d}.npz", indices)
    if "gradcam" in engines:
        for mode in ("context", "image_only"):
            _check_array_sample(
                directory / "xai" / "gradcam_img" /
                f"gradcam_arrays_{mode}_epoch_{int(row['best_epoch']):03d}.npz", indices)
    if "lore" in engines:
        epoch = int(row["best_epoch"])
        lore = pd.read_csv(directory / f"xai/lore_sa/epoch{epoch:03d}/lore_sa_persample_epoch{epoch:03d}.csv")
        if (not np.array_equal(lore.val_index.to_numpy(), indices)
                or not np.array_equal(lore.row_id.astype(str), xs.stamp_row_ids(indices)["row_id"])
                or not np.array_equal(lore.true_label.to_numpy(),
                                      xf._CTX.validation_df.target.to_numpy()[indices])):
            raise RuntimeError(f"LORE sample identity mismatch at {directory}")


def _permutation(model, row, directory, sample):
    raw, images, labels = sample
    attribution = directory / "xai" / "shap_tab" / "aligned_attributions.npz"
    if not attribution.exists():
        original = xf._logit_np
        counts = dict(evaluated=0, clipped=0)

        def observed_logit(probability, eps=1e-6):
            values = np.asarray(probability)
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite probability during gamma Permutation SHAP")
            counts["evaluated"] += int(values.size)
            counts["clipped"] += int(((values <= eps) | (values >= 1 - eps)).sum())
            return original(probability, eps=eps)

        xf._logit_np = observed_logit
        try:
            phi, meta = xf.aligned_tab_attributions(model, raw, images, background="baseline")
        finally:
            xf._logit_np = original
        if phi.shape != (len(raw), len(xf._CTX.feat_cols)) or not np.isfinite(phi).all():
            raise RuntimeError(f"Invalid Permutation attribution array at {directory}")
        temporary = attribution.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary, phi=phi, val_index=xf.FAITH_SUBSET_IDX, labels=labels,
            **xs.stamp_row_ids(xf.FAITH_SUBSET_IDX))
        _write_json(attribution.with_suffix(".json"), {**meta, **counts})
        temporary.replace(attribution)
    _check_array_sample(attribution, xf.FAITH_SUBSET_IDX)
    with np.load(attribution) as saved:
        phi = saved["phi"]
    xf.explanation_faithfulness(
        model, str(directory), int(row["best_epoch"]),
        methods=("tabular", "image", "joint"), tab_imp_override=np.abs(phi),
        out_dir=str(directory / "xai" / "faithfulness"), verbose=False)


def _signed(model, row, directory, sample):
    raw, images, labels = sample
    with np.load(directory / "xai" / "shap_tab" / "aligned_attributions.npz") as saved:
        phi = saved["phi"]
    xf.aligned_signed_faithfulness(
        model, str(directory), int(row["best_epoch"]), raw_df=raw, imgs=images,
        labels=labels, phi=phi, subset_idx=xf.FAITH_SUBSET_IDX,
        modes=xf.SIGNED_MODES, out_dir=str(directory / "xai" / "shap_tab"),
        q_maxes=(0.20, 0.60), resume=True, verbose=False)


def _gpu(model, row, directory, sample):
    indices = xf.FAITH_SUBSET_IDX
    _stage(directory, "shap", ("xai/shap_tab/epoch_*/*.npz",),
           lambda: _callbacks(model, row, directory, ("shap",), indices))
    _stage(directory, "gradcam", ("xai/gradcam_img/gradcam_arrays_context_*.npz",
                                 "xai/gradcam_img/gradcam_arrays_image_only_*.npz"),
           lambda: _callbacks(model, row, directory, ("gradcam",), indices))
    _stage(directory, "permutation", ("xai/shap_tab/aligned_attributions.npz",
                                     "xai/shap_tab/aligned_attributions.json",
                                     "xai/faithfulness/faithfulness_curves.npz",
                                     "xai/faithfulness/faithfulness.json"),
           lambda: _permutation(model, row, directory, sample))
    # The virtual run points the gradient reader to this gamma's SHAP, never the source run.
    _stage(directory, "gradient_faithfulness", ("xai/faith_gradient/faithfulness_curves.npz",
                                              "xai/faith_gradient/faithfulness.json"),
           lambda: xf.explanation_faithfulness(
               model, str(directory), int(row["best_epoch"]), methods=("tabular", "joint"),
               out_dir=str(directory / "xai" / "faith_gradient"), verbose=False))
    signed_files = tuple(f"xai/shap_tab/faithfulness_aligned_signed_{mode}{suffix}"
                         for mode in xf.SIGNED_MODES for suffix in (".json", "_curves.npz"))
    _stage(directory, "signed", signed_files,
           lambda: _signed(model, row, directory, sample))


def _plan(art_path, study_dir):
    art_path, study_dir = Path(art_path).resolve(), Path(study_dir).resolve()
    if study_dir.parent != art_path / "runs" / "controlled_fusion":
        raise ValueError("Select the canonical Controlled Fusion study under this artifact root")
    master = pd.read_csv(art_path / "summaries" / "master_results_rows.csv")
    master = master.loc[master.impl_name.eq("controlled_fusion")].copy()
    if len(master) != 6 or not master.seed.is_unique or not master.run_id.is_unique:
        raise ValueError("The canonical fusion registry must contain six distinct seeds and runs")
    master["source_run_dir"] = [paths.run_dir_abs(d, art_path) for d in master.run_dir]
    if any(Path(d).resolve().parents[1] != study_dir for d in master.source_run_dir):
        raise ValueError("Fusion registry and selected canonical study differ")
    gamma = pd.read_csv(study_dir / "controlled_fusion_posthoc_gamma_summary.csv")
    if set(gamma.run_id) != set(master.run_id) or gamma.duplicated(["run_id", "delta"]).any():
        raise ValueError("The gamma grid does not match the canonical checkpoint registry")
    expected = np.arange(-0.5, 0.501, 0.125)
    for _, group in gamma.groupby("run_id"):
        if len(group) != len(expected) or not np.allclose(sorted(group.delta), expected):
            raise ValueError("Every checkpoint must have the complete nine-point gamma grid")
    if not (np.allclose(gamma.gamma_img, 1 - gamma.delta)
            and np.allclose(gamma.gamma_tab, 1 + gamma.delta)):
        raise ValueError("Gamma coefficients disagree with their recorded delta")
    if not np.isfinite(gamma[["val_auc", "brier", "ece", "img_drop_auc",
                              "tab_drop_auc", "modality_balance"]].to_numpy()).all():
        raise ValueError("Complete the saved predictive gamma grid before gamma XAI")
    if xf.FAITH_SUBSET_IDX is None:
        raise RuntimeError("Initialize the adopted XAI sample with @@SETUP_XAI@@ first")
    indices = np.asarray(xf.FAITH_SUBSET_IDX, dtype=int)
    if not xf.ACTIVE_SUBSET_FINGERPRINT or not np.array_equal(
            xd.SHAP_KWARGS.get("subset_idx"), indices):
        raise RuntimeError("Initialize the adopted XAI sample with @@SETUP_XAI@@ first")
    lore_indices = None
    for row in master.itertuples(index=False):
        cfg = json.loads((Path(row.source_run_dir) / "config.json").read_text())
        xd.assert_run_feature_space(cfg, row.source_run_dir, xf._CTX.feature_space_file,
                                    xf._CTX.feature_space_fingerprint)
        if float(cfg["gamma_img"]) != 1 or float(cfg["gamma_tab"]) != 1:
            raise ValueError(f"Expected a neutral fitted checkpoint: {row.run_id}")
        source = Path(row.source_run_dir) / "xai" / "lore_sa" / f"epoch{int(row.best_epoch):03d}"
        lore = pd.read_csv(source / f"lore_sa_persample_epoch{int(row.best_epoch):03d}.csv")
        actual = lore.val_index.to_numpy(dtype=int)
        if (len(np.unique(actual)) != len(actual) or not np.isin(actual, indices).all()
                or not np.array_equal(lore.row_id.astype(str), xs.stamp_row_ids(actual)["row_id"])):
            raise ValueError(f"Invalid canonical nested LORE sample: {row.run_id}")
        if lore_indices is not None and not np.array_equal(lore_indices, actual):
            raise ValueError("The canonical fusion LORE samples differ between seeds")
        lore_indices = actual
    gamma = gamma.drop(columns=["seed", "setting_tag", "run_dir", "base_model_name"], errors="ignore")
    plan = master[["run_id", "seed", "source_run_dir", "best_epoch", "model_name"]].merge(
        gamma, on="run_id", validate="one_to_many").rename(
            columns={"run_id": "source_run_id", "model_name": "base_model_name"})
    plan["setting_tag"] = [f"d{d:+.3f}".replace(".", "p").replace("+", "plus").replace("-", "minus")
                           for d in plan.delta]
    plan["run_id"] = plan.source_run_id + "@" + plan.setting_tag
    plan["impl_name"] = "controlled_fusion"
    return plan.sort_values(["seed", "delta"]), lore_indices


def _controls(rows):
    neutral = next(row for row in rows if row["delta"] == 0)
    base = Path(neutral["output_dir"]) / "xai"
    with np.load(base / "shap_tab" / "aligned_attributions.npz") as data:
        phi_neutral = data["phi"]
    base_meta = json.loads((base / "shap_tab" / "aligned_attributions.json").read_text())
    result = []
    for row in rows:
        current = Path(row["output_dir"]) / "xai"
        meta = json.loads((current / "shap_tab" / "aligned_attributions.json").read_text())
        with np.load(current / "shap_tab" / "aligned_attributions.npz") as data:
            error = np.max(np.abs(data["phi"] - row["gamma_tab"] * phi_neutral))
        record = {"run_id": row["run_id"], "source_run_id": row["source_run_id"],
                  "seed": row["seed"], "delta": row["delta"],
                  "permutation_scale_max_abs_error": float(error),
                  "permutation_clipped_scores": meta["clipped"],
                  "neutral_permutation_clipped_scores": base_meta["clipped"],
                  "permutation_scaling_unclipped": meta["clipped"] == base_meta["clipped"] == 0}
        for mode in ("context", "image_only"):
            name = f"gradcam_arrays_{mode}_epoch_{int(row['best_epoch']):03d}.npz"
            with np.load(base / "gradcam_img" / name) as a, np.load(current / "gradcam_img" / name) as b:
                p, q = a["preds"].reshape(-1), b["preds"].reshape(-1)
                eligible = (p > 1e-6) & (p < 1 - 1e-6) & (q > 1e-6) & (q < 1 - 1e-6)
                record[f"cam_{mode}_unclipped_n"] = int(eligible.sum())
                record[f"cam_{mode}_max_abs_delta_unclipped"] = (
                    float(np.max(np.abs(a["cams"][eligible] - b["cams"][eligible])))
                    if eligible.any() else np.nan)
        result.append(record)
    return result


def _summaries(plan, root):
    files, lore_rows = [], []
    for row in plan.to_dict("records"):
        directory = Path(row["output_dir"])
        curves = [("canonical", "abs", directory / "xai/faithfulness/faithfulness_curves.npz"),
                  ("gradient", "abs", directory / "xai/faith_gradient/faithfulness_curves.npz")]
        curves.extend(("aligned_signed", mode, directory / "xai/shap_tab" /
                       f"faithfulness_aligned_signed_{mode}_curves.npz") for mode in xf.SIGNED_MODES)
        for source, mode, path in curves:
            _check_array_sample(path, xf.FAITH_SUBSET_IDX)
            files.append({**{k: row[k] for k in ("run_id", "impl_name", "seed", "setting_tag")},
                          "source": source, "mode": mode, "npz_path": str(path)})
        epoch = int(row["best_epoch"])
        lore = pd.read_csv(directory / f"xai/lore_sa/epoch{epoch:03d}/lore_sa_persample_epoch{epoch:03d}.csv")
        covered = lore.neigh_n_classes.fillna(0).ge(2)
        lore_rows.append(dict(run_id=row["run_id"], lore_coverage=float(covered.mean()),
                              lore_rule_length=float(lore.loc[covered, "rule_len"].mean()),
                              lore_cv_bacc=float(lore.loc[covered, "fid_cv_bacc"].mean()),
                              lore_unique_fraction=float(lore.neigh_unique_frac.mean()),
                              lore_attempts=len(lore), lore_covered=int(covered.sum()),
                              lore_errors=int(lore.error.notna().sum())))
    strata = xf.stratified_table(str(root), files=pd.DataFrame(files), verbose=False)
    metadata = plan[["run_id", "source_run_id", "delta", "gamma_img", "gamma_tab"]]
    strata = strata.merge(metadata, on="run_id", validate="many_to_one")
    strata.to_csv(root / "faithfulness_strata.csv", index=False)
    all_rows = strata.loc[strata.stratum.eq("all")].copy()
    all_rows["metric"] = [f"{s}_{m}_{k}_{d}_q{int(round(q * 100))}"
                          for s, m, k, d, q in zip(all_rows.source, all_rows["mode"],
                                                  all_rows.method, all_rows.direction, all_rows.q_max)]
    wide = all_rows.pivot(index="run_id", columns="metric", values="gain_norm").reset_index()
    per_seed = plan.merge(wide, on="run_id", validate="one_to_one").merge(
        pd.DataFrame(lore_rows), on="run_id", validate="one_to_one")
    per_seed.to_csv(root / "cf_xai_per_seed.csv", index=False)
    metrics = ["val_auc", "brier", "ece", "img_drop_auc", "tab_drop_auc", "modality_balance",
               *wide.columns.drop("run_id"), *pd.DataFrame(lore_rows).columns.drop("run_id")]
    summary = per_seed.groupby(["delta", "gamma_img", "gamma_tab"], sort=True).agg(
        n_seeds=("seed", "nunique"),
        **{f"{key}_{stat}": (key, stat) for key in metrics for stat in ("mean", "std")}).reset_index()
    summary.to_csv(root / "cf_xai_summary.csv", index=False)
    return summary


def run_cf_gamma_xai(art_path, study_dir, *, pilot=False):
    """Produce the full frozen-gamma XAI grid; resume verified completed stages.

    Setup through @@SETUP_XAI@@ must already adopt the canonical XAI sample.
    The nested LORE sample and predictive grid are read from existing study outputs.
    A pilot uses the first seed at neutral and both extremes. Full production evaluates
    all nine settings, including a fresh seeded neutral for internally matched LORE.
    """
    import keras

    plan, lore_indices = _plan(art_path, study_dir)
    if pilot:
        plan = plan.loc[plan.seed.eq(plan.seed.min()) & plan.delta.isin([-.5, 0, .5])].copy()
    root = Path(art_path).resolve() / "interventions" / "controlled_fusion" / Path(study_dir).name
    plan["output_dir"] = [str(root / f"seed_{int(s)}" / tag)
                          for s, tag in zip(plan.seed, plan.setting_tag)]
    methods = {name: _sha256(Path(__file__).with_name(name + ".py")) for name in
               ("cf_xai", "xai_faithfulness", "xai_engines", "xai_driver", "xai_lore_sa", "xai_subset")}
    common = dict(
        protocol="frozen-gamma-xai-v1", source_hashes=methods,
        versions={name: version(name) for name in ("keras", "tensorflow", "shap", "lore-sa", "numpy")},
        feature_space=xf._CTX.feature_space_fingerprint,
        sample_fingerprint=xf.ACTIVE_SUBSET_FINGERPRINT,
        sample_indices=np.asarray(xf.FAITH_SUBSET_IDX).tolist(),
        lore_indices=lore_indices.tolist(), lore_config=xd.lore_sa_config(),
        shap_config={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in xd.SHAP_KWARGS.items()},
        gradcam_config=dict(engine=xd.GRADCAM_KWARGS, callback=xd.GRADCAM_CB_KWARGS,
                            modes=list(xd.GRADCAM_MODES), batch_size=xd.GRADCAM_BATCH),
        faithfulness=dict(budgets=list(xf.FAITH_BUDGETS), random_seed=xf.FAITH_RNG_SEED,
                          random_draws=xf.FAITH_N_RANDOM, blur_sigma=xf.FAITH_BLUR_SIGMA,
                          image_replace=xf.FAITH_IMG_REPLACE, insertion=xf.FAITH_DO_INSERTION,
                          permutation_seed=0, background="baseline"))
    for source_dir, group in plan.groupby("source_run_dir", sort=False):
        checkpoints = list((Path(source_dir) / "checkpoints").glob("best_*.keras"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected exactly one selected checkpoint under {source_dir}")
        digest = _sha256(checkpoints[0])
        for row in group.to_dict("records"):
            directory = Path(row["output_dir"])
            protocol = {**common, **{k: row[k] for k in ("source_run_id", "seed", "delta",
                                                       "gamma_img", "gamma_tab")},
                        "checkpoint_sha256": digest}
            path = directory / "protocol.json"
            if path.exists():
                if json.loads(path.read_text()) != protocol:
                    raise RuntimeError(f"Intervention protocol changed; existing outputs are protected: {path}")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(f"Unidentified intervention outputs: {directory}")
                directory.mkdir(parents=True, exist_ok=True)
                _write_json(path, protocol)
    print(f"[CF XAI] {'PILOT' if pilot else 'FULL GRID'}: {len(plan)} settings; {root}", flush=True)
    sample = xf._build_faith_subset()
    encoded = xf._encode_raw(sample[0])
    controls = []
    for source_dir, group in plan.groupby("source_run_dir", sort=False):
        if all(_complete(Path(d), ("shap", "gradcam", "permutation", "gradient_faithfulness", "signed"))
               for d in group.output_dir):
            controls.extend(_controls(group.to_dict("records")))
            continue
        model, _ = xd.load_run_model(source_dir)
        try:
            fusion = model.get_layer("fusion_with_gamma")
            fusion.gamma_img.assign(1.)
            fusion.gamma_tab.assign(1.)
            canonical = Path(source_dir) / "xai/faithfulness/faithfulness_curves.npz"
            _check_array_sample(canonical, xf.FAITH_SUBSET_IDX)
            with np.load(canonical) as saved:
                np.testing.assert_allclose(xf._score_logit(model, encoded, sample[1]),
                                           saved["s_orig"], rtol=1e-4, atol=1e-3,
                                           err_msg="Neutral checkpoint predictions do not reproduce")
            ordered = group.assign(_distance=group.delta.abs()).sort_values(["_distance", "delta"])
            for row in ordered.to_dict("records"):
                fusion.gamma_img.assign(float(row["gamma_img"]))
                fusion.gamma_tab.assign(float(row["gamma_tab"]))
                _gpu(model, row, Path(row["output_dir"]), sample)
            controls.extend(_controls(group.to_dict("records")))
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()
    pd.DataFrame(controls).to_csv(root / ("pilot_structural_controls.csv" if pilot else "structural_controls.csv"), index=False)
    previous_lore, previous_device = xd.LORE_SA_KWARGS.copy(), xd.LORE_SA_DEVICE
    try:
        xd.LORE_SA_DEVICE = "/CPU:0"
        xd.LORE_SA_KWARGS.update(subset_idx=lore_indices, max_explain_samples=len(lore_indices),
                                 max_samples=int(lore_indices.max()) + 1, max_batches=None)
        for source_dir, group in plan.groupby("source_run_dir", sort=False):
            if all(_complete(Path(d), ("lore",)) for d in group.output_dir):
                continue
            model, _ = xd.load_run_model(source_dir, device="/CPU:0")
            try:
                fusion = model.get_layer("fusion_with_gamma")
                for row in group.to_dict("records"):
                    fusion.gamma_img.assign(float(row["gamma_img"]))
                    fusion.gamma_tab.assign(float(row["gamma_tab"]))
                    directory = Path(row["output_dir"])
                    _stage(directory, "lore", ("xai/lore_sa/epoch*/lore_sa_persample_*.csv",),
                           lambda: _callbacks(model, row, directory, ("lore",), lore_indices))
            finally:
                del model
                keras.backend.clear_session()
                gc.collect()
    finally:
        xd.LORE_SA_KWARGS.clear()
        xd.LORE_SA_KWARGS.update(previous_lore)
        xd.LORE_SA_DEVICE = previous_device
    destination = root / "pilot" if pilot else root
    destination.mkdir(parents=True, exist_ok=True)
    summary = _summaries(plan, destination)
    print(f"[CF XAI] complete: {len(plan)} settings; summaries under {destination}", flush=True)
    return summary
