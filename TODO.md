# Project TODO

> Tasks are organized by **phase** (sequential dependency) and labeled by **priority**:
> `[CRITICAL]` must be done for the project to function · `[HIGH]` significant value · `[MEDIUM]` important but deferrable · `[LOW]` nice-to-have / research directions
---

## Foundation

- [ ] `[C]` Migrate all notebook logic to production-ready code
- [ ] `[C]` Implement MLflow tracking (params, metrics, dataset hash, git tag)

## Experimentation

- [ ] `[C]` Run full experiment suite on server + analyze all runs in MLflow
- [ ] `[H]` Test pipeline on multiclass problem (currently binary)

## Explainability

- [ ] `[L]` Add additional visual XAI method (smoothgrad or Grad-CAM++)
- [ ] `[H]` Modality weight analysis — test impact on discrimination + explanation quality
- [ ] `[H]` Per-sample confidence & uncertainty output for clinician ⚠️ *ask Pellungrini on metric choice*
- [ ] `[M]` Zone-based hybrid ensemble (decision space clustering + learning-to-defer)

## Interaction & Deployment

- [ ] `[M]` Interactive what-if interface — modality weight sliders → live prediction update
- [ ] `[M]` Model perturbation at deployment + inference correction demo

---

## Open

- [ ] Decide: remote vs local MLflow server
- [ ] Decide: multiclass target definition + evaluation metric
- [ ] Decide: deployment format (API / notebook demo / clinical UI)

---

## Completed

<!-- Move tasks here with completion date once done -->
- [x] `[M]` Update readme file -- completed 2026-03-01
- [x] `[H]` Add tabular XAI method (Lore from XAI-Lib [https://github.com/kdd-lab/XAI-Lib.git]) 2026-27-01

