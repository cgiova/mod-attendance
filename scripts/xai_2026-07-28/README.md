# Harness di verifica — sessione 28 luglio 2026

Script one-off che hanno prodotto i numeri citati nei memo di
`artifacts/enhanced/reports/`. Vivevano nella scratchpad di sessione (`/tmp`); sono
qui perché i memo non avessero risultati senza percorso di riproduzione — lo stesso
problema già registrato per la FeatureSpace non deterministica.

**Non sono entry point supportati.** Tutti tranne `nbcheck.py`/`freenames.py`/`run_strata_signed.py`
fanno `exec` delle celle di `notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb`
fino alla prima che contiene `xf.configure`, per ricostruire i globals del notebook
(`ART_PATH`, `feature_space`, `training_df`, …). Se quel notebook viene riordinato,
vanno riadattati. Il codice riutilizzabile sta in `src/isic24/xai_*.py`.

| script | cosa produce | memo |
|---|---|---|
| `run_strata_signed.py` | `runs/_faithfulness/faithfulness_strata_signed{,_pooled}.csv` — **CPU, ~20 s, nessun notebook**: sola re-integrazione delle 24 curve signed a q≤0.2 e q≤0.6, con `gain_norm`. Auto-verifica P1–P5. | `FINDINGS_budget_signed_2026-07-30.md` |
| `run_signed.py` | `runs/_faithfulness/faithfulness_aligned_signed.csv` + `runs/_robustness/aligned_signed/<run_id>/` — 1 run per famiglia, n=128 | `FINDINGS_signed_tabular_2026-07-28.md` |
| `run_signed_a05.py` | idem, ma forza `fixed_scaling` seed 47 setting `a0p500` (**non ancora eseguito**, vedi sotto) | idem |
| `equiv.py` | equivalenza numerica notebook-vecchio vs `src/isic24/xai_*` — PASS 6/6 famiglie, scarto 0 | — |
| `test_signed.py` | unit test dell'attributore allineato (efficienza di Permutation, `Σφ = logit(p(x)) − logit(p(base))`) | `FINDINGS_signed_tabular_2026-07-28.md` |
| `test_driver.py` | smoke test di `run_posthoc_xai` (overwrite on/off, niente righe duplicate nei CSV in append) | — |
| `bench_lore.py` + `bench_lore.csv` | benchmark `lore_explainer` vs `lore_sa`, genetic/random, min/sample — solo **tempo** | `BENCH_lore_2026-07-28.md` |
| `bench_lore_quality.py` | sceglie il generatore del vicinato: diversità, `neigh_pos_rate`, `rule_len`, **`fid_cv_bacc`** | `BENCH_lore_2026-07-28.md` (Addendum) |
| `cell_fill_missing_shap.py` | la stessa cosa come **cella da `%load`** nel notebook post-hoc, sul contesto già caricato | — |
| `fill_missing_shap.py` | genera gli array SHAP dei **20** run che non li hanno (tutti `learnable_scaling` con `init_alpha≠0.5`) — additivo, ~30 min | — |
| `nbcheck.py`, `freenames.py` | checker AST dei nomi liberi nei notebook | — |

La stratificazione `abs` (`FINDINGS_strata_2026-07-28.md`) **non** ha uno script qui: sta
tutta in `src/isic24/xai_faithfulness.py` (`build_strata`, `stratified_table`,
`pool_by_family`, `verify_reintegration`). Quella signed sì — è `run_strata_signed.py`,
che è anche l'unico script della cartella eseguibile senza GPU e senza il notebook.

## Due guard, aggiunti il 2026-07-29

**Non sovrascrivere le run canoniche.** `xai_driver.assert_writable(run_dir, engines)` rifiuta
**per engine**, in base agli artefatti che quell'engine ha già su disco:

| engine | percorsi guardati | run con artefatti |
|---|---|---|
| `shap` | `xai/shap/epoch_*`, `xai/shap/*.csv` | 90/110 |
| `gradcam` | `xai/gradcam/epoch_*` | 90/110 |
| `lore` | `xai/lore/epoch*`, `xai/lore/*.csv` | 90/110 |

Riempire quello che manca resta permesso — è additivo, ed è a cosa serve
`run_posthoc_xai_batch`. Il vecchio `overwrite=False` **non** proteggeva niente: salta solo in
presenza di `xai/xai_status.json`, che scrive il driver nuovo e che nessuna delle 110 run
canoniche ha.

`overwrite` e `allow_clobber` sono knob **diversi**, di proposito:

| | significato | cosa disattiva |
|---|---|---|
| `overwrite=True` | «non saltare, ricalcola» | la scorciatoia di `xai_status.json` |
| `allow_clobber=True` | «puoi distruggere quello che c'è» | `assert_writable` |

Quindi `overwrite=True` da solo **solleva comunque** se ci sono artefatti. Ricalcolare in una
dir vuota è ordinaria amministrazione, riscrivere una run che ha già prodotto i suoi artefatti
no, e le due cose non devono condividere la stessa parola. Bypass globale, se proprio serve:
`xai_driver.CANONICAL_WRITE_GUARD = False`.

**Fuori scope del guard:** `xai/faithfulness/` (`faithfulness.json`, `faithfulness_curves.npz`)
non lo scrive il driver ma `explanation_faithfulness`, ed è governato dal knob `FAITH_OVERWRITE`
del loop post-hoc, che di suo riusa il JSON esistente. Quello resta come prima.

**Non cambiare il set su cui si misura.** `isic24.eval_guard.assert_canonical_eval(NB)` alza
`RuntimeError` se `EVAL_SET != "canonical"` o `FAITH_N_SAMPLES != 128`, e controlla lo snapshot
`validation_ids.SEED49.json` con la stessa politica della cella 16 (warn finché lo snapshot è
una rigenerazione post-hoc, raise se viene da un training run). I knob restano usabili per gli
esperimenti esplorativi — che però devono scrivere altrove, non negli artefatti dello studio
primario. Chiamato da `run_signed.py`, `bench_lore.py` e dalla cella P1 del notebook di prova.

## Prima di lanciarli

`notebooks/modality_attendance/isic24_modality_attendance_smoke.ipynb` contiene celle
di prova corte (P0–P8) che verificano un pezzo per volta della stessa catena: GPU, setup,
deserializzazione dei checkpoint, efficienza di Permutation, pipeline signed a n=8,
driver post-hoc. Vanno eseguite prima dei run lunghi.

`build_smoke_nb.py` è il generatore di quel notebook. **Rigenerandolo lo sovrascrive**, output
delle celle compresi: se modifichi il notebook a mano, non rilanciare il builder (o riporta
prima la modifica qui).

## In sospeso

`run_signed_a05.py` non è mai stato eseguito. La riga `fixed_scaling` in
`faithfulness_aligned_signed.csv` è `a0p000`, cioè l'endpoint immagine-only in cui il
ramo tabellare è scalato a zero: tutti i modi valgono esattamente 0. È un controllo
valido ma non un valore rappresentativo della famiglia. Da rilanciare **su GPU**:

    python scripts/xai_2026-07-28/run_signed_a05.py

(seleziona da sé il run, quindi `PER_FAMILY` qui non ha effetto; verificato che la
stringa che rimpiazza in `run_signed.py` combacia ancora.)

Tutte le altre righe del CSV sono state calcolate su Quadro RTX 6000; se un giorno
questa venisse calcolata su CPU, va annotato nel memo — la tabella serve a confrontare
famiglie fra loro e presuppone lo stesso substrato.
