# Notebook archiviati — non sono codice vivo

Nessuno di questi viene eseguito e nessuno va modificato. Stanno qui, invece che cancellati,
perché sono la storia di come si è arrivati alla pipeline attuale; gli output sono stati
rimossi (pesavano più del codice).

**La pipeline viva è `notebooks/modality_attendance/isic24_modality_attendance_{train,posthoc}.ipynb`**,
più i moduli in `src/isic24/`.

| file | cos'era |
|---|---|
| `isic24_modality_attendance_MONOLITE.ipynb` | il notebook unico prima della divisione train/post-hoc del 2026-07-28 |
| `isic24_modality_attendance_framework.ipynb` | la versione "framework" che l'ha preceduto |
| `isic24_modality_attendance_framework_colab.ipynb` | la stessa, adattata a Colab |
| `train_2026-07-30_sera.ipynb`, `posthoc_2026-07-30_sera.ipynb`, `posthoc_PRE_2026-07-30.ipynb` | backup manuali di una serata di lavoro, prima che i backup li facesse git |
| `train_enhanced.ipynb` | stava dentro `src/isic24/`, cioè dentro il package: un notebook non è codice importabile |

Le versioni con output, e ogni stato precedente, restano nella storia di git:
`git log --follow -- <path>` e `git show <commit>:<path>`.
