"""Full-validation gamma evaluation from a saved study, without notebook or XAI state."""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths


def _validation_dataset(art_root, cfg, model):
    import h5py
    import keras
    import tensorflow as tf
    from .xai_driver import (
        assert_fs_matches_model, feature_space_fingerprint,
    )

    frame = pd.read_parquet(art_root / "evaluation_frame.parquet")
    snapshot = json.loads((art_root / cfg["validation_snapshot"]).read_text())
    row_ids = frame[snapshot["row_id_col"]].astype(str).tolist()
    assert row_ids == list(map(str, snapshot["row_ids"])), "Validation snapshot mismatch"
    assert len(frame) == cfg["n_val"], "Validation size mismatch"
    fs = keras.models.load_model(art_root / cfg["feature_space_file"])
    _, fingerprint = feature_space_fingerprint(fs)
    if cfg.get("feature_space_fingerprint") is not None:
        assert fingerprint == cfg["feature_space_fingerprint"], "FeatureSpace mismatch"
    assert_fs_matches_model(model, fs, cfg["run_id"])
    raw = {
        col: (frame[col].to_numpy() if pd.api.types.is_numeric_dtype(frame[col].dtype)
              else frame[col].astype(str).to_numpy())
        for col in fs.features
    }
    encoded = np.asarray(fs(raw))
    image_shape = next(tuple(t.shape[1:]) for t in model.inputs if len(t.shape) == 4)
    assert image_shape[-1] == 3, "Expected RGB model input"
    image_file = paths.data_dir(dataset=cfg["dataset"].split("/")[0]) / "train-image.hdf5"
    with h5py.File(image_file, "r") as store:
        images = [store[row_id][()] for row_id in row_ids]
    labels = frame.target.to_numpy(dtype=np.float32).reshape(-1, 1)
    assert set(np.unique(labels)) == {0, 1}, "Expected both binary validation classes"

    def decode(image, features, label):
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_shape[:2], method="area")
        image = tf.ensure_shape(tf.cast(image, tf.float32) / 255.0, image_shape)
        return {"images": image, "features": features}, label

    ds = tf.data.Dataset.from_tensor_slices((images, encoded, labels))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).cache()
    return ds.batch(int(cfg["batch_size"])).prefetch(tf.data.AUTOTUNE), image_shape


def _metrics(model, dataset, include_attendance=False):
    import keras

    auc = keras.metrics.AUC()
    if include_attendance:
        import tensorflow as tf
        image_off_auc, table_off_auc = keras.metrics.AUC(), keras.metrics.AUC()
    labels, probabilities = [], []
    for x, y in dataset:
        prediction = model(x, training=False)
        auc.update_state(y, prediction)
        if include_attendance:
            image_off = dict(x, images=tf.ones_like(tf.cast(x["images"], tf.float32)) * 0.5)
            table_off = dict(x, features=tf.zeros_like(tf.cast(x["features"], tf.float32)))
            image_off_auc.update_state(y, model(image_off, training=False))
            table_off_auc.update_state(y, model(table_off, training=False))
        labels.append(np.asarray(y).reshape(-1))
        probabilities.append(np.asarray(prediction).reshape(-1))
    y, p = np.concatenate(labels), np.concatenate(probabilities)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities"
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & ((p <= hi) if i == 14 else (p < hi))
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    metrics = {"val_auc": float(auc.result().numpy()),
               "brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}
    if include_attendance:
        image_drop = metrics["val_auc"] - float(image_off_auc.result().numpy())
        table_drop = metrics["val_auc"] - float(table_off_auc.result().numpy())
        metrics.update(img_drop_auc=image_drop, tab_drop_auc=table_drop,
                       modality_balance=table_drop / (image_drop + table_drop + 1e-8))
    return metrics


def complete_gamma_calibration(study_dir, *, include_attendance=False):
    """Complete saved gamma rows, or evaluate the standard nine-point grid if absent.

    Read frozen validation and FeatureSpace identities from the selected study. Preserve
    measured values, validate saved AUCs, and update study/per-run CSVs and seed aggregates.
    Optionally complete modality AUC drops using the training replacement convention.
    """
    import keras
    import keras_cv  # Registers the backbone serializers used by saved checkpoints.
    from . import mod_attendance  # Registers the shared fusion layers.

    study_dir = Path(study_dir).expanduser().resolve()
    assert study_dir.parent.name == "controlled_fusion", "Select a Controlled Fusion study"
    art_root = study_dir.parents[2]
    runs = pd.read_csv(study_dir / "combined_summary.csv", skipinitialspace=True)
    runs.columns = runs.columns.str.strip()
    for col in runs.select_dtypes(include=["object", "string"]):
        runs[col] = runs[col].str.strip()
    assert len(runs) and runs.run_id.is_unique, "Empty or duplicate study runs"
    assert runs.impl_name.eq("controlled_fusion").all(), "Wrong strategy"
    configs, directories = {}, {}
    for row in runs.itertuples(index=False):
        run_dir = Path(paths.run_dir_abs(row.run_dir, art_root))
        assert run_dir.parents[1] == study_dir, f"Run outside selected study: {run_dir}"
        cfg = json.loads((run_dir / "config.json").read_text())
        assert cfg["run_id"] == row.run_id and cfg["seed"] == row.seed, "Run identity mismatch"
        assert cfg["impl_name"] == "controlled_fusion", "Wrong checkpoint strategy"
        directories[row.run_id], configs[row.run_id] = run_dir, cfg
    first = next(iter(configs.values()))
    for key in ("dataset", "feature_space_file", "feature_space_fingerprint",
                "validation_snapshot", "n_val", "batch_size"):
        assert len({cfg.get(key) for cfg in configs.values()}) == 1, f"Mixed study {key}"

    csv_path = study_dir / "controlled_fusion_posthoc_gamma_summary.csv"
    if csv_path.exists():
        table = pd.read_csv(csv_path, skipinitialspace=True)
        table.columns = table.columns.str.strip()
    else:
        records = []
        for row in runs.itertuples(index=False):
            for delta in np.round(np.arange(-0.5, 0.501, 0.125), 3):
                tag = f"d{delta:+.3f}".replace(".", "p").replace("+", "plus").replace("-", "minus")
                records.append(dict(
                    run_id=row.run_id, setting_tag=row.setting_tag, seed=int(row.seed),
                    run_dir=str(directories[row.run_id]), posthoc_id=tag, delta=float(delta),
                    gamma_img=float(1 - delta), gamma_tab=float(1 + delta),
                    mode=("tab_enhance_img_penalize" if delta > 0 else
                          "tab_penalize_img_enhance" if delta < 0 else "neutral"),
                    base_model_name=row.model_name,
                ))
        table = pd.DataFrame(records)
    assert set(table.run_id) == set(runs.run_id), "Sweep/study run mismatch"
    assert not table.duplicated(["run_id", "posthoc_id"]).any(), "Duplicate gamma rows"
    assert np.allclose(table.gamma_img, 1 - table.delta), "Image gamma/delta mismatch"
    assert np.allclose(table.gamma_tab, 1 + table.delta), "Table gamma/delta mismatch"
    metrics = ["val_auc", "brier", "ece"]
    if include_attendance:
        metrics += ["img_drop_auc", "tab_drop_auc", "modality_balance"]
    for metric in metrics:
        if metric not in table:
            table[metric] = np.nan

    dataset, image_shape = None, None
    for run_id, rows in table.groupby("run_id", sort=False):
        missing = rows[metrics].isna().any(axis=1)
        if not missing.any():
            continue
        cfg = configs[run_id]
        checkpoint = directories[run_id] / "checkpoints" / f"best_{cfg['model_name']}.keras"
        model = keras.models.load_model(checkpoint, compile=False, safe_mode=False)
        try:
            if dataset is None:
                dataset, image_shape = _validation_dataset(art_root, first, model)
            assert next(tuple(t.shape[1:]) for t in model.inputs if len(t.shape) == 4) == image_shape
            fusion = model.get_layer("fusion_with_gamma")
            fusion.gamma_img.assign(1.0)
            fusion.gamma_tab.assign(1.0)
            neutral = _metrics(model, dataset, include_attendance)
            assert abs(neutral["val_auc"] - cfg["base_auc"]) <= 5e-4, (
                f"Neutral AUC mismatch for {run_id}; check validation data and checkpoint"
            )
            for idx, row in rows.loc[missing].iterrows():
                fusion.gamma_img.assign(float(row.gamma_img))
                fusion.gamma_tab.assign(float(row.gamma_tab))
                values = neutral if row.delta == 0 else _metrics(model, dataset, include_attendance)
                if pd.notna(row.val_auc):
                    assert abs(values["val_auc"] - row.val_auc) <= 5e-4, (
                        f"Saved sweep AUC mismatch for {run_id}/{row.posthoc_id}"
                    )
                for metric, value in values.items():
                    if pd.isna(row[metric]):
                        table.loc[idx, metric] = value
            print(f"[CF CALIBRATION] {run_id}: completed {int(missing.sum())} gamma rows")
        finally:
            del model
            gc.collect()

    assert np.isfinite(table[metrics].to_numpy()).all(), "Incomplete metrics"
    table = table.sort_values(["seed", "gamma_tab"]).reset_index(drop=True)
    provenance_path = csv_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance.update(n_runs=len(runs), dataset=first["dataset"])
    provenance["calibration"] = dict(
        population="full_validation", n_val=first["n_val"], ece_n_bins=15,
        validation_snapshot=first["validation_snapshot"],
        feature_space_file=first["feature_space_file"],
        feature_space_fingerprint=first.get("feature_space_fingerprint"),
        run_ids=list(configs), deltas=sorted(table.delta.unique().tolist()),
    )
    if include_attendance:
        provenance["attendance"] = dict(
            population="full_validation", image_replacement=0.5, encoded_table_replacement=0.0,
            balance="tab_drop_auc / (img_drop_auc + tab_drop_auc + 1e-8)",
        )
    provenance_path.write_text(json.dumps(provenance, indent=2))
    table.to_csv(csv_path, index=False)
    for run_id, rows in table.groupby("run_id", sort=False):
        diagnostics = directories[run_id] / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        rows.to_csv(diagnostics / "posthoc_gamma_summary.csv", index=False)
    aggregations = dict(
        mean=("val_auc", "mean"), std=("val_auc", "std"), count=("val_auc", "count"),
        brier_mean=("brier", "mean"), brier_std=("brier", "std"),
        ece_mean=("ece", "mean"), ece_std=("ece", "std"),
    )
    for metric in ("img_drop_auc", "tab_drop_auc", "modality_balance"):
        if metric in table:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_std"] = (metric, "std")
    aggregate = table.groupby(["delta", "gamma_img", "gamma_tab"]).agg(**aggregations).reset_index()
    aggregate.to_csv(study_dir / "controlled_fusion_posthoc_gamma_aggregated.csv", index=False)
    return aggregate
