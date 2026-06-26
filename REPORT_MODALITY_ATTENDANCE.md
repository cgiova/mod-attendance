# Report progetto: modality attendance framework

## Obiettivo
Questo progetto studia come bilanciare in modo esplicito due sorgenti di informazione in un modello multimodale per ISIC 2024:

1. immagini della lesione
2. feature tabellari cliniche e derivate

L'obiettivo non e` solo massimizzare la performance predittiva, ma capire se le strategie di modality attendance migliorano anche:

- stabilita del modello
- calibrazione delle probabilita
- dipendenza effettiva dalle due modalita
- leggibilita e fedelta delle spiegazioni

Il punto importante emerso dagli esperimenti e` che `validation AUC` misura bene la performance, ma non misura direttamente la qualita delle spiegazioni. Per la spiegabilita oggi sono presenti proxy quantitativi utili ma incompleti.

## Dati e setup sperimentale
- Dataset ISIC 2024 multimodale con immagini e metadati tabellari.
- Split: `StratifiedGroupKFold` a 5 fold con grouping per `patient_id`, per evitare leakage tra train e validation.
- Target estremamente sbilanciato all'origine: circa `99.9%` negativi e `0.1%` positivi.
- Sampling nel notebook: negativi downsampled e positivi upsampled, arrivando a circa `66.6%` negativi e `33.4%` positivi nel training.
- Backbone immagine: `EfficientNetV2`.
- Feature tabellari: preprocessing con `keras.utils.FeatureSpace`.
- Loss: binary cross entropy con label smoothing.
- Metrica di training principale: `val_auc`.
- Esperimenti multi-seed principali: seed `47`, `53`, `71`, `8`, `90`, salvo dove indicato.

## Metriche usate

### Metriche predittive
- `best_val_auc`: migliore AUC di validazione lungo il training.
- `base_auc`: AUC del checkpoint scelto, calcolata sul validation set.
- `brier`: media dell'errore quadratico tra probabilita predetta ed etichetta vera; piu` basso e` meglio.
- `ece`: scarto medio pesato tra accuratezza e confidenza per bin di probabilita; piu` basso e` meglio.
- `inference_ms_per_batch`: tempo medio di inferenza per batch.

### Metriche di dipendenza dalle modalita
- `img_drop_auc`: calo di AUC quando l'immagine viene ablated/sostituita.
- `tab_drop_auc`: calo di AUC quando le feature tabellari vengono ablated/azzerate.
- `modality_balance`: rapporto derivato dai drop; valori vicini a `0` indicano comportamento piu` image-driven, valori vicini a `1` comportamento piu` tabular-driven.

Queste metriche sono molto utili per capire quale modalita sostiene la predizione, ma non misurano direttamente se una spiegazione locale e` buona.

### Proxy quantitativi per le spiegazioni
Nel notebook sono presenti tre famiglie XAI:

- `SHAP`, applicato alla parte tabellare encoded.
- `Grad-CAM`, applicato al ramo immagine.
- `LORE`, per regole locali tabellari.

Le metriche quantitative effettivamente salvate per SHAP sono:

- `total_abs_shap`: massa totale media di importanza assoluta.
- `top1_share`: quota di importanza concentrata nella feature encoded piu` importante.
- `top5_share`: quota di importanza concentrata nelle 5 feature encoded piu` importanti.
- `shap_entropy`: calcolata in analisi di sweep quando disponibili gli array SHAP.
- `shap_fidelity_r2`: regressione lineare tra somma dei valori SHAP e predizione del modello.

Interpretazione corretta:

- `top1_share`, `top5_share` e `shap_entropy` misurano concentrazione o dispersione dell'importanza, non la correttezza causale della spiegazione.
- `total_abs_shap` misura intensita dell'attribuzione, ma non dice se l'attribuzione identifica davvero le parti decisive.
- `shap_fidelity_r2` e` piu` vicino alla fedelta, ma resta limitato: usa solo SHAP tabellare e verifica una relazione lineare con l'output, non una controfattualita tramite perturbazione.
- Grad-CAM e LORE producono artefatti interpretabili, ma non hanno ancora una metrica quantitativa di bonta comparabile tra esperimenti.

Per `adaptive_gating` e` presente anche:

- `spearman_w_img_delta_img`
- `spearman_w_tab_delta_tab`

Queste sono metriche di gate faithfulness: misurano se il peso del gate correla con l'impatto reale della modalita quando viene ablated. Sono utili per validare il gate, non per validare direttamente SHAP, Grad-CAM o LORE.

## Risultati complessivi

| Famiglia | n run | mean best AUC | std | mean Brier | mean ECE | Lettura sintetica |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Vanilla | 5 | 0.901 | 0.012 | 0.151 | 0.140 | Baseline solida e abbastanza bilanciata |
| Fixed scaling | 45 | 0.896 | 0.021 | 0.155 | 0.141 | Sweep ampio; migliora quando aumenta il peso tabellare |
| Learnable scaling | 5 | 0.893 | 0.017 | 0.171 | 0.165 | Sensibile all'inizializzazione; poco convincente |
| Loss regularization | 25 | 0.896 | 0.009 | 0.163 | 0.156 | Non migliora stabilmente la baseline |
| Controlled fusion | 5 | 0.894 | 0.007 | 0.153 | 0.143 | Fusione controllata senza vantaggio netto |
| Adaptive gating | 5 | 0.912 | 0.014 | 0.119 | 0.087 | Variante piu` promettente su AUC e calibrazione |

Nota: `fixed_scaling` include tutti gli alpha e quindi la media globale e` penalizzata dai casi image-only. La lettura corretta e` per valore di `alpha`, non solo sulla media aggregata.

## Esperimenti

### 1. Vanilla multimodale
Architettura:

- ramo immagine con `EfficientNetV2` + pooling + dropout
- ramo tabellare con MLP
- concatenazione finale
- classificatore sigmoid

Risultati su 5 seed:

- mean `best_val_auc`: `0.901`
- std `best_val_auc`: `0.012`
- best run: seed `53`, AUC `0.9190`
- mean `brier`: `0.151`
- mean `ece`: `0.140`
- mean `img_drop_auc`: `0.083`
- mean `tab_drop_auc`: `0.078`
- mean `modality_balance`: `0.481`

Interpretazione:

- La baseline usa entrambe le modalita in modo abbastanza bilanciato.
- I drop medi di AUC sono simili per immagine e tabella, ma con variabilita tra seed.
- La calibrazione non e` ottimale: ECE medio intorno a `0.140`.
- E` una baseline credibile per confrontare ogni strategia di attendance.

### 2. Fixed scaling
Architettura:

- embeddings immagine e tabellare calcolate separatamente
- peso tabellare fisso `alpha`
- peso immagine fisso `beta = 1 - alpha`
- concatenazione delle embedding pesate

Sweep:

| alpha tabellare | mean AUC | mean Brier | mean ECE | mean modality_balance | mean img_drop | mean tab_drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.850 | 0.160 | 0.124 | 0.000 | 0.350 | 0.000 |
| 0.125 | 0.881 | 0.170 | 0.164 | 0.251 | 0.151 | 0.050 |
| 0.250 | 0.904 | 0.160 | 0.158 | 0.256 | 0.139 | 0.047 |
| 0.375 | 0.893 | 0.170 | 0.166 | 0.367 | 0.114 | 0.066 |
| 0.500 | 0.901 | 0.158 | 0.151 | 0.375 | 0.099 | 0.059 |
| 0.625 | 0.898 | 0.160 | 0.150 | 0.491 | 0.083 | 0.084 |
| 0.750 | 0.908 | 0.158 | 0.156 | 0.557 | 0.074 | 0.089 |
| 0.875 | 0.912 | 0.147 | 0.136 | 0.689 | 0.054 | 0.121 |
| 1.000 | 0.914 | 0.112 | 0.060 | 1.000 | 0.000 | 0.414 |

Best single run:

- `alpha = 0.75`
- seed `53`
- AUC `0.9209`

Interpretazione:

- Il trend piu` chiaro e` l'aumento di performance spostando peso verso la tabella.
- `alpha = 1.0` e` il migliore in media per AUC, Brier ed ECE, ma diventa di fatto tabular-only.
- `alpha = 0.875` e` un compromesso interessante: molto tabellare, ma non cancella completamente il ramo immagine.
- `alpha = 0.75` produce il miglior run singolo, pero` non e` il migliore in media.
- Questa famiglia dimostra che nel validation split il segnale tabellare e` molto forte.

### 3. Learnable scaling
Architettura:

- singolo parametro trainabile `alpha = sigmoid(logit)`
- `beta = 1 - alpha`
- scaling simmetrico delle due embedding

Sweep su `init_alpha`: `0.1`, `0.3`, `0.5`, `0.7`, `0.9`.

| init_alpha | best AUC | Brier | ECE | modality_balance | img_drop | tab_drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.873 | 0.181 | 0.180 | 0.466 | 0.116 | 0.101 |
| 0.3 | 0.880 | 0.170 | 0.158 | 0.304 | 0.113 | 0.049 |
| 0.5 | 0.894 | 0.167 | 0.162 | 0.479 | 0.088 | 0.081 |
| 0.7 | 0.908 | 0.159 | 0.145 | 0.400 | 0.107 | 0.071 |
| 0.9 | 0.910 | 0.179 | 0.181 | 0.773 | 0.053 | 0.179 |

Risultato medio:

- mean `best_val_auc`: `0.893`
- mean `brier`: `0.171`
- mean `ece`: `0.165`

Interpretazione:

- L'inizializzazione alta verso la tabella produce AUC migliore.
- La calibrazione resta peggiore rispetto a fixed scaling tabellare e adaptive gating.
- Il parametro learnable tende a restare vicino all'inizializzazione, quindi non sembra imparare autonomamente un bilanciamento robusto.
- La variante e` metodologicamente interessante, ma oggi non e` la piu` forte.

### 4. Loss regularization
Architettura:

- modello multimodale tipo vanilla
- penalizzazione sui pesi finali associati alla componente tabellare
- sweep su `lambda_reg`

| lambda_reg | mean AUC | mean Brier | mean ECE | mean modality_balance |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.899 | 0.165 | 0.163 | 0.518 |
| 1e-5 | 0.897 | 0.170 | 0.166 | 0.537 |
| 1e-4 | 0.896 | 0.162 | 0.151 | 0.459 |
| 1e-3 | 0.892 | 0.160 | 0.156 | 0.486 |
| 1e-2 | 0.894 | 0.157 | 0.144 | 0.518 |

Best single run:

- `lambda_reg = 0`
- seed `8`
- AUC `0.9097`

Interpretazione:

- La penalizzazione non migliora la performance rispetto alla baseline senza regolarizzazione.
- Aumentare `lambda_reg` puo` ridurre leggermente Brier/ECE in alcuni casi, ma non in modo sufficiente da compensare la perdita di AUC.
- Il controllo e` troppo indiretto: agisce sull'ultimo layer, non sulla causalita dell'uso delle modalita.

### 5. Controlled fusion
Architettura:

- proiezione immagine e tabella nello stesso spazio
- fusione additiva con `gamma_img` e `gamma_tab`
- nella versione corrente i gamma sono inizializzati a `1` e non trainabili

Risultati su 5 seed:

- mean `best_val_auc`: `0.894`
- std `best_val_auc`: `0.007`
- best run: seed `8`, AUC `0.9050`
- mean `brier`: `0.153`
- mean `ece`: `0.143`
- mean `img_drop_auc`: `0.068`
- mean `tab_drop_auc`: `0.081`
- mean `modality_balance`: `0.538`

Interpretazione:

- Non supera vanilla.
- Il comportamento medio e` leggermente piu` tabular-oriented della baseline.
- Il nome "controlled fusion" e` piu` ambizioso dell'implementazione attuale: senza una vera procedura di tuning/inference sui gamma, il controllo post-training resta limitato.

### 6. Adaptive gating
Architettura:

- embedding immagine e tabellare portate entrambe a dimensione `256`
- rete di gating che produce due pesi `softmax`: `w_img`, `w_tab`
- fusione additiva pesata sample-wise
- penalita di entropia per evitare gate troppo degeneri

Risultati su 5 seed:

- mean `best_val_auc`: `0.912`
- std `best_val_auc`: `0.014`
- best run: seed `71`, AUC `0.9286`
- mean `brier`: `0.119`
- mean `ece`: `0.087`
- mean `img_drop_auc`: `0.020`
- mean `tab_drop_auc`: `0.106`
- mean `modality_balance`: `0.853`

Diagnostica gate:

- mean `gate_weight_mean_img`: `0.365`
- mean `gate_weight_mean_tab`: `0.635`
- mean `spearman_w_img_delta_img`: `0.559`
- mean `spearman_w_tab_delta_tab`: `0.530`

Interpretazione:

- E` la famiglia piu` promettente in termini di compromesso tra AUC e calibrazione.
- La calibrazione migliora molto rispetto a vanilla: ECE medio `0.087` contro `0.140`.
- L'ablation mostra che la tabella resta la modalita piu` decisiva.
- La correlazione Spearman gate/impatto e` positiva e moderata: il gate contiene informazione utile sull'importanza della modalita, ma non e` una spiegazione perfetta.
- In un seed l'ablation immagine produce `img_drop_auc` leggermente negativo: questo segnala che rimuovere l'immagine puo` perfino migliorare la AUC in quel run, confermando che l'informazione tabellare domina.

## Analisi delle spiegazioni generate

### SHAP tabellare
SHAP e` la parte piu` quantitativamente tracciata.

Proxy medi salvati da `shap_summary.csv`, usando l'ultima riga disponibile per ogni run:

| Famiglia | n summary | total_abs_shap | top1_share | top5_share |
| --- | ---: | ---: | ---: | ---: |
| Vanilla | 10 | 0.293 | 0.082 | 0.275 |
| Fixed scaling | 54 | 0.345 | 0.073 | 0.245 |
| Learnable scaling | 1 | 0.193 | 0.053 | 0.242 |
| Loss regularization | 25 | 0.281 | 0.085 | 0.280 |
| Controlled fusion | 5 | 0.407 | 0.073 | 0.264 |
| Adaptive gating | 5 | 0.720 | 0.076 | 0.257 |

Lettura:

- `adaptive_gating` ha massa SHAP tabellare piu` alta, coerente con il fatto che il modello usa molto la tabella.
- `top5_share` e` relativamente simile tra molte famiglie, quindi non emerge una differenza enorme nella concentrazione top-5.
- `loss_regularization` ha `top5_share` medio leggermente piu` alto, quindi le attribuzioni risultano piu` concentrate, ma questo non significa automaticamente che siano piu` fedeli.
- `fixed_scaling` include anche casi estremi; quando `alpha = 0`, SHAP tabellare puo` essere nullo o quasi nullo per costruzione.

Limite fondamentale:

- Queste metriche dicono come sono distribuite le attribuzioni, non se rimuovere le feature indicate da SHAP cambia davvero la predizione piu` di una rimozione casuale.

### Grad-CAM
Grad-CAM produce griglie e triptychs per ispezionare:

- immagine originale con predizione
- heatmap
- overlay

Limite attuale:

- non viene calcolata una metrica quantitativa su Grad-CAM.
- non ci sono `deletion/insertion`, `pointing game`, IoU con maschere, o confronto con regioni clinicamente annotate.

Senza maschere di segmentazione o annotazioni cliniche, la strada piu` naturale e` una metrica perturbation-based: rimuovere progressivamente le regioni piu` salienti e misurare quanto cala lo score del modello.

### LORE
LORE e` progettato per produrre regole locali tabellari e frequenze delle feature nelle regole.

Limite attuale:

- in alcune run, soprattutto per `adaptive_gating`, la callback LORE fallisce con mismatch di shape.
- finche questo bug non viene corretto, LORE non deve essere usato come evidenza comparativa forte.
- manca una metrica di fidelity locale delle regole, ad esempio accuratezza della regola nel vicinato sintetico o concordanza regola-modello.

## Cosa manca per valutare davvero la bonta delle spiegazioni
Al momento manca una metrica principale di explanation quality confrontabile tra strategie.

Le metriche presenti sono proxy:

- `SHAP concentration`: utile per capire dispersione/concentrazione.
- `SHAP fidelity R2`: utile ma limitato alla relazione lineare SHAP-output.
- `Gate faithfulness`: utile per il gate, non per SHAP/Grad-CAM/LORE.
- `Modality ablation`: utile per capire quale modalita conta, non quale spiegazione e` corretta.

Quindi oggi si puo` dire:

- quale modello predice meglio
- quale modello e` piu` calibrato
- quale modalita e` piu` importante
- se i gate correlano moderatamente con l'impatto delle modalita
- come si distribuiscono le attribuzioni SHAP

Non si puo` ancora dire in modo forte:

- "questa strategia produce spiegazioni migliori"
- "questa Grad-CAM e` piu` fedele"
- "questa attention/importanza locale identifica davvero le evidenze causali"

## Metrica consigliata per la bonta delle spiegazioni
La metrica che consiglio di introdurre come principale e` una `Explanation-Guided Deletion AUC`, cioe` una curva di ablation guidata dalla spiegazione.

Idea:

1. Per ogni sample di validation si calcola la spiegazione.
2. Si ordinano feature tabellari o regioni immagine per importanza decrescente.
3. Si rimuove progressivamente il top `k%` dell'evidenza indicata dalla spiegazione.
4. Si misura quanto cala lo score del modello.
5. Si confronta il calo con una rimozione casuale dello stesso budget.

Formula pratica in logit space:

- `s(x) = logit(p_model(x))` per la classe predetta o per la classe positiva.
- `x_del(q)` = input con top `q` di evidenza rimossa secondo la spiegazione.
- `x_rand(q)` = input con lo stesso budget rimosso casualmente.
- `C_exp(q) = s(x) - s(x_del(q))`
- `C_rand(q) = s(x) - s(x_rand(q))`
- `Faithfulness(q) = C_exp(q) - C_rand(q)`
- `ExplanationFaithfulnessAUC = area_q Faithfulness(q)`

Interpretazione:

- valore alto: la spiegazione identifica parti che, se rimosse, danneggiano davvero la predizione piu` del caso.
- valore vicino a zero: la spiegazione non e` meglio di una rimozione casuale.
- valore negativo: la spiegazione e` potenzialmente fuorviante.

Perche questa metrica e` adatta a questo progetto:

- non richiede maschere di segmentazione o annotazioni umane.
- funziona sia su tabella sia su immagine.
- e` direttamente legata al comportamento del modello.
- permette di confrontare vanilla, fixed scaling, learnable scaling, loss regularization, controlled fusion e adaptive gating sulla stessa validation subset.
- risponde alla domanda centrale: le strategie di attendance migliorano o peggiorano la fedelta delle spiegazioni?


## Conclusione
Il risultato sperimentale piu` forte e` che la componente tabellare domina questo validation split. `Fixed scaling` mostra chiaramente che aumentare il peso della tabella migliora la performance media, mentre `adaptive_gating` offre il miglior compromesso complessivo per AUC e calibrazione.

Sul piano XAI, pero`, il progetto non ha ancora una metrica definitiva di bonta delle spiegazioni. Le metriche SHAP attuali sono utili come proxy descrittivi; la gate faithfulness valida parzialmente il comportamento del gate; la modality ablation misura la dipendenza dalle modalita. Nessuna di queste, da sola, dimostra che le spiegazioni locali siano fedeli.

La prossima estensione metodologica dovrebbe quindi essere una metrica perturbation-based, in particolare `Explanation-Guided Deletion AUC` con baseline casuale. E` la scelta piu` coerente con quanto gia` implementato e permette di rispondere in modo quantitativo alla domanda: le strategie di attendance migliorano davvero la qualita delle spiegazioni, oppure migliorano solo la AUC?
