# CMMD attendance sweep cells

These replacements extend the selected CMMD studies. The notebooks are edited by pasting
the blocks below; the shared Python implementations are integrated independently. Run the
training setup with the saved split, sampling, FeatureSpace and six seeds unchanged.

## Training notebook

In `notebooks/modality_attendance/cmmd/cmmd_modattendance_train.ipynb`, replace the start of
`@@SWEEP_GATING@@`, stopping immediately before `ag_all_results = []`, with:

```python
# @@SWEEP_GATING@@ — Adaptive gating entropy sweep over the existing six seeds
EPOCHS = CFG.epochs
SEEDS = CFG.seeds
ADAPTIVE_GATING_SETTINGS = [
    {"tag": "ent0e+00", "entropy_weight": 0.0},
    {"tag": "base", "entropy_weight": 1e-4},
    {"tag": "ent1e-03", "entropy_weight": 1e-3},
    {"tag": "ent1e-02", "entropy_weight": 1e-2},
    {"tag": "ent1e-01", "entropy_weight": 1e-1},
]
IMPL = "adaptive_gating"
STUDY_TAG = "20261020-020650"
study_tag, study_dir = open_study(IMPL, STUDY_TAG)
```

Keep the remaining training loop. The `base` tag preserves the six existing checkpoint
identities: the runner skips these and trains the remaining 24 configurations. Resume the
same study timestamp after interruptions. Replace the `ag_agg = ...` assignment with:

```python
ag_agg = ag_combined.groupby("entropy_weight").agg(
    mean=("best_val_auc", "mean"), std=("best_val_auc", "std"),
    count=("best_val_auc", "count"),
    brier_mean=("brier", "mean"), brier_std=("brier", "std"),
    ece_mean=("ece", "mean"), ece_std=("ece", "std"),
    img_drop_mean=("img_drop_auc", "mean"), tab_drop_mean=("tab_drop_auc", "mean"),
    balance_mean=("modality_balance", "mean"),
).reset_index()
```

Keep `@@SWEEP_CF@@` at its single neutral training setting and selected timestamp
`20261020-063421`. Replace the complete `@@CF_GAMMA_CALIBRATION@@` cell with:

```python
# @@CF_GAMMA_CALIBRATION@@ — complete predictive and attendance metrics on the frozen gamma grid
from isic24.fusion_calibration import complete_gamma_calibration

display(complete_gamma_calibration(study_dir, include_attendance=True))
```

Execute this cell immediately after `@@SWEEP_CF@@`, before another sweep reassigns
`study_dir`. Completed CF training runs are reused. The evaluation fills missing AUC,
Brier, ECE, image/table AUC drops and modality balance on nine frozen-weight settings.
Saved measurements are preserved. Image removal uses constant 0.5 and encoded table removal
uses zeros, matching training diagnostics. Modality balance retains the signed training
formula; it is not clipped and can fall outside [0, 1] if an ablation improves AUC.

## Post-hoc notebook

In `notebooks/modality_attendance/cmmd/cmmd_modattendance_posthoc.ipynb`, keep `CANONICAL`
unchanged. In `@@PARAMETERS@@`, replace the block starting at `N_CANONICAL_RUNS = 132`
and ending immediately before `cf_study_dir = ...` with:

```python
TRAINING_SEEDS = [8, 47, 53, 71, 90, 101]
EXPECTED_SETTINGS = {
    "vanilla": 1, "fixed_scaling": 9, "learnable_scaling": 5,
    "loss_regularization": 5, "controlled_fusion": 1, "adaptive_gating": 5,
}
N_CANONICAL_RUNS = len(TRAINING_SEEDS) * sum(EXPECTED_SETTINGS.values())
_runs = {i: sorted(glob.glob(os.path.join(d, "*", "*", "config.json")))
         for i, d in CANONICAL.items()}
_configs = {c: json.load(open(c)) for cs in _runs.values() for c in cs}
_spaces = {c: cfg.get("feature_space_file") for c, cfg in _configs.items()}
_space_fps = {c: cfg.get("feature_space_fingerprint") for c, cfg in _configs.items()}
for _impl, _cs in _runs.items():
    _cfgs = pd.DataFrame([_configs[c] for c in _cs])
    assert len(_cfgs) == len(TRAINING_SEEDS) * EXPECTED_SETTINGS[_impl], (
        f"{_impl}: incomplete training grid; finish the selected training sweep first"
    )
    assert _cfgs.run_id.is_unique and _cfgs.best_val_auc.notna().all(), (
        f"{_impl}: duplicate or unfinished runs"
    )
    assert _cfgs.setting_tag.nunique() == EXPECTED_SETTINGS[_impl], _impl
    assert not _cfgs.duplicated(["setting_tag", "seed"]).any(), _impl
    assert _cfgs.groupby("setting_tag").seed.apply(
        lambda values: set(values) == set(TRAINING_SEEDS)
    ).all(), f"{_impl}: seed grid mismatch"
    if _impl == "adaptive_gating":
        _weights = _cfgs.groupby("setting_tag").entropy_weight
        assert _weights.nunique().eq(1).all(), "Mixed entropy weights within a setting"
        assert set(_weights.first()) == {0.0, 1e-4, 1e-3, 1e-2, 1e-1}, "Entropy grid mismatch"
        assert _weights.first().loc["base"] == 1e-4, "Wrong reference entropy weight"
    print(f"{_impl}: {len(_cs)} completed runs, {EXPECTED_SETTINGS[_impl]} settings")

assert (sum(map(len, _runs.values())) == N_CANONICAL_RUNS
        and None not in _spaces.values() and None not in _space_fps.values()
        and len(set(_space_fps.values())) == 1), (
    f"Expected {N_CANONICAL_RUNS} runs with one FeatureSpace fingerprint"
)
```

The trained registry has 156 rows: CF's 54 intervention evaluations stay in its dedicated
sweep CSV, outside the trained-run registry. `CENTRAL_SETTINGS` remains unchanged;
Adaptive Gating's `base` row still means entropy weight 1e-4. SHAP, Grad-CAM, LORE,
faithfulness, signed evidence and shape loops already iterate over the trained registry,
so they include all 30 gating runs. Existing completion checks reuse matching outputs.

In `@@GATE_DIAGNOSTICS@@`, replace the `_TRAINED = ...` through `assert ...` block,
stopping immediately before `all_diag = []`, with:

```python
_TRAINED = ["base_auc", "img_drop_auc", "tab_drop_auc", "brier", "ece"]
_NEEDED = {"run_id", "run_dir", "setting_tag", "seed", "entropy_weight", *_TRAINED}
assert _NEEDED <= set(ag_combined.columns), f"Missing gate columns: {_NEEDED - set(ag_combined.columns)}"
assert ag_combined.run_id.is_unique, "Duplicate gate runs"
assert len(ag_combined) == len(TRAINING_SEEDS) * EXPECTED_SETTINGS["adaptive_gating"], (
    "Incomplete adaptive-gating study aggregate; rerun the updated training sweep"
)
assert not ag_combined.duplicated(["setting_tag", "seed"]).any(), "Duplicate setting/seed"
assert ag_combined.groupby("setting_tag").seed.apply(
    lambda values: set(values) == set(TRAINING_SEEDS)
).all(), "Gate seed grid mismatch"
assert set(ag_combined.entropy_weight) == {0.0, 1e-4, 1e-3, 1e-2, 1e-1}, "Gate entropy grid mismatch"
```

The existing `xl.gate_summary(gate)` call now also exports per-observation gate-weight SD,
mean/SD entropy in nats, and the fraction with maximum modality weight at least 0.9. These
statistics distinguish input-dependent selection from a constant allocation with the same
mean. Diagnostics cover full validation and remain separate from the balanced XAI sample.

Replace the complete `@@CF_GAMMA_SWEEP@@` cell with:

```python
# @@CF_GAMMA_SWEEP@@ — complete frozen gamma metrics and plot the saved evaluations
from isic24.fusion_calibration import complete_gamma_calibration

posthoc_agg = complete_gamma_calibration(cf_study_dir, include_attendance=True)
posthoc_all_df = pd.read_csv(os.path.join(cf_study_dir, "controlled_fusion_posthoc_gamma_summary.csv"))
for _, row in cf_combined.iterrows():
    cf_run_dir = xs.run_dir_abs(row["run_dir"], ART_PATH)
    posthoc_df = posthoc_all_df[posthoc_all_df.run_id == row.run_id].sort_values("gamma_tab")
    plots_dir = os.path.join(cf_run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(posthoc_df["gamma_tab"], posthoc_df["val_auc"], marker="o")
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("gamma_tab")
    plt.ylabel("val_auc")
    plt.title(f"Controlled fusion post-hoc: val_auc vs gamma_tab (seed {int(row['seed'])})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "posthoc_val_auc_vs_gamma_tab.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(posthoc_df["gamma_img"], posthoc_df["gamma_tab"], c=posthoc_df["val_auc"])
    for _, r in posthoc_df.iterrows():
        plt.text(r["gamma_img"], r["gamma_tab"], f"{r['val_auc']:.3f}", fontsize=8)
    plt.xlabel("gamma_img")
    plt.ylabel("gamma_tab")
    plt.title(f"Controlled fusion post-hoc sweep (seed {int(row['seed'])})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "posthoc_gamma_sweep_scatter.png"), dpi=300)
    plt.close()

display(posthoc_agg)
```

The CF evaluation function is the same in training and post-hoc. Completed metrics require
no further model inference. The existing two per-run AUC figures are retained. Frozen gamma
evaluations do not create independently trained models or additional rows in `master`.

CF explanation producers still use the neutral checkpoint. Explaining every frozen gamma
requires separate intervention-scoped attribution/rule outputs; this predictive sweep does
not relabel neutral explanations as gamma-specific results.

## Execution order

1. Run training setup with the existing frozen protocol. Run `@@SWEEP_CF@@` to select and
   reuse its six fitted checkpoints, followed by `@@CF_GAMMA_CALIBRATION@@`.
2. Run the updated `@@SWEEP_GATING@@`. Complete all 30 run identities before post-hoc setup.
3. Run post-hoc setup through `@@STOP@@`, including `@@SETUP_XAI@@`, then `@@MASTER_TABLE@@`.
4. Run the existing explanation producers and `@@GATE_DIAGNOSTICS@@` on the extended registry.
   `@@CF_GAMMA_SWEEP@@` reuses the saved CF metrics and draws its existing figures.

Code and cell checks use synthetic inputs with mocked writes. Training, checkpoint inference
and experimental outputs await user execution.
