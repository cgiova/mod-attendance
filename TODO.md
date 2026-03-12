# Project TODO

> Tasks are organized by **phase** (sequential dependency) and labeled by **priority**:
> `[CRITICAL]` must be done for the project to function · `[HIGH]` significant value · `[MEDIUM]` important but deferrable · `[LOW]` nice-to-have / research directions
---

## Phase 1 · Foundation

- [x] `[M]` Update readme file
- [ ] `[C]` Migrate all notebook logic to production-ready code
- [ ] `[C]` Implement MLflow tracking (params, metrics, dataset hash, git tag)

## Phase 2 · Experimentation

- [ ] `[C]` Run full experiment suite on server + analyze all runs in MLflow
- [ ] `[H]` Test pipeline on multiclass problem (currently binary)

## Phase 3 · Explainability & Trustworthiness

- [ ] `[H]` Add visual XAI method (e.g. Integrated gradients)
- [ ] `[H]` Modality weight analysis — test impact on discrimination + explanation quality
- [ ] `[H]` Per-sample confidence & uncertainty output for clinician ⚠️ *ask Pellegrini on metric choice*
- [ ] `[M]` Zone-based hybrid ensemble (decision space clustering + learning-to-defer)

## Phase 4 · Interaction & Deployment

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
<!-- - [x] Task name — completed YYYY-MM-DD -->