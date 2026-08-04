"""L'identita' di una run: un posto solo, entrambe le direzioni.

Il framework scrive la stessa run in DUE codifiche:

    run_dir = <runs>/<impl_name>/<exp_tag>/<setting_folder>
    run_id  = <exp_tag>__<setting_folder>

dove::

    exp_tag        = "<study_tag>"  oppure  "<study_tag>/seed_<seed>"
    setting_folder = "<setting_tag>__e<epochs>"   (con le degenerazioni sotto)

``run_dir`` separa con ``/`` e porta ``impl_name``; ``run_id`` fonde gli ultimi due livelli
con ``__`` e ``impl_name`` non c'e'.  Quando ``exp_tag`` contiene il seed, ``run_id`` finisce
per avere **un** ``/`` e **due** ``__``, e il secondo ``__`` appartiene al nome del setting:
per questo non e' ri-parsabile a occhio, ed e' il bug che il 2026-07-30 mandava tutte le run
nel gruppo ``"?"`` collassando le sei famiglie in una riga di medie.

**Questo modulo e' la definizione**, e rispecchia `run_experiment` del notebook di training
(``_make_exp_tag`` / ``_make_setting_folder``) incluse le sue degenerazioni:

===============  ==========  ==========================
``setting_tag``  ``epochs``  ``setting_folder``
===============  ==========  ==========================
``"a0p000"``     ``20``      ``"a0p000__e20"``
``None``         ``20``      ``"run__e20"``
``"a0p000"``     ``None``    ``"a0p000"``
``None``         ``None``    ``"run"``
===============  ==========  ==========================

e il fatto che ``exp_tag`` generato in automatico e' il solo timestamp, **senza** livello
``seed_``: una run lanciata fuori da un loop multi-seed vive a tre livelli sotto ``runs/``,
non quattro.  `parse_run_dir` riconosce entrambi i layout.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

SEED_DIR_RE = re.compile(r"^seed_(?P<seed>.+)$")
EPOCHS_SUFFIX_RE = re.compile(r"^(?P<setting_tag>.*)__e(?P<epochs>\d+)$")
NO_SETTING = "run"          # cosa scrive `_make_setting_folder` quando setting_tag e' None


# --------------------------------------------------------------------------------------
# Costruzione -- le formule, in un posto solo
# --------------------------------------------------------------------------------------

def make_exp_tag(study_tag=None, seed=None) -> str:
    """``study_tag`` (default: timestamp di adesso), piu' ``/seed_<seed>`` se c'e' un seed."""
    tag = datetime.now().strftime("%Y%m%d-%H%M%S") if study_tag is None else str(study_tag)
    return f"{tag}/seed_{seed}" if seed is not None else tag


def make_setting_folder(setting_tag, epochs) -> str:
    """Il nome della cartella di setting. Degenerazioni comprese (vedi tabella in testa)."""
    parts = [str(setting_tag)] if setting_tag else [NO_SETTING]
    if epochs is not None:
        parts.append(f"e{int(epochs)}")
    return "__".join(parts)


def make_run_id(exp_tag, setting_folder) -> str:
    """Il ``run_id`` canonico dai due pezzi che il training ha gia' in mano."""
    return f"{exp_tag}__{setting_folder}"


def make_run_dir(runs_path, impl_name, exp_tag, setting_folder) -> str:
    """La ``run_dir`` canonica. ``exp_tag`` puo' contenere ``/``: os.path.join lo gestisce."""
    return os.path.join(str(runs_path), str(impl_name), str(exp_tag), str(setting_folder))


def run_id_from_parts(study_tag, seed, setting_tag, epochs) -> str:
    """Scorciatoia: dai cinque campi al ``run_id``, senza passare per i due pezzi."""
    return make_run_id(make_exp_tag(study_tag, seed), make_setting_folder(setting_tag, epochs))


# --------------------------------------------------------------------------------------
# Lettura -- l'inversa che mancava
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    """I campi di una run. ``seed`` ed ``epochs`` sono ``None`` dove il layout non li ha."""
    impl_name: str
    study_tag: str
    seed: str | None
    setting_tag: str | None
    epochs: int | None

    @property
    def exp_tag(self) -> str:
        return make_exp_tag(self.study_tag, self.seed)

    @property
    def setting_folder(self) -> str:
        return make_setting_folder(self.setting_tag, self.epochs)

    @property
    def run_id(self) -> str:
        return make_run_id(self.exp_tag, self.setting_folder)


def parse_run_dir(run_dir, runs_path=None) -> RunKey:
    """Dai path ai campi, riconoscendo sia il layout a 4 livelli sia quello a 3.

    Con ``runs_path`` il taglio e' esatto.  Senza, ci si ancora alla coda e si decide dal
    penultimo livello: se combacia con ``seed_<...>`` c'e' il livello di seed, altrimenti no.
    L'euristica serve perche' i CSV del registro portano i path assoluti della macchina su
    cui si e' allenato, che qui non esistono.
    """
    parts = [p for p in os.path.normpath(str(run_dir)).split(os.sep) if p]

    if runs_path is not None:
        root = [p for p in os.path.normpath(str(runs_path)).split(os.sep) if p]
        if parts[:len(root)] != root:
            raise ValueError(f"{run_dir!r} non sta sotto {runs_path!r}")
        rest = parts[len(root):]
        if len(rest) < 3:
            raise ValueError(f"attesi almeno impl/exp_tag/setting sotto {runs_path!r}: {rest}")
        impl_name, setting_folder = rest[0], rest[-1]
        exp_parts = rest[1:-1]
    else:
        if len(parts) < 3:
            raise ValueError(f"run_dir troppo corto: {run_dir!r}")
        has_seed = len(parts) >= 4 and SEED_DIR_RE.match(parts[-2]) is not None
        n = 4 if has_seed else 3
        impl_name, setting_folder = parts[-n], parts[-1]
        exp_parts = parts[-n + 1:-1]

    seed = None
    if exp_parts and (m := SEED_DIR_RE.match(exp_parts[-1])):
        seed, exp_parts = m.group("seed"), exp_parts[:-1]
    if not exp_parts:
        raise ValueError(f"nessuno study_tag in {run_dir!r} (exp_tag ridotto al solo seed)")
    study_tag = "/".join(exp_parts)

    epochs = None
    setting_tag = setting_folder
    if m := EPOCHS_SUFFIX_RE.match(setting_folder):
        setting_tag, epochs = m.group("setting_tag"), int(m.group("epochs"))
    if setting_tag == NO_SETTING:
        setting_tag = None

    return RunKey(impl_name=impl_name, study_tag=study_tag, seed=seed,
                  setting_tag=setting_tag, epochs=epochs)


def run_id_from_run_dir(run_dir, runs_path=None) -> str:
    """Il ``run_id`` di una run, dedotto dal suo path."""
    return parse_run_dir(run_dir, runs_path).run_id


def run_id_from_artifact(path, marker="xai", runs_path=None) -> str:
    """``run_id`` da un file DENTRO la run (es. ``<run_dir>/xai/lore/.../x.csv``)."""
    parts = os.path.normpath(str(path)).split(os.sep)
    if marker not in parts:
        raise ValueError(f"{marker!r} non compare in {path!r}: impossibile isolare la run dir")
    cut = len(parts) - 1 - parts[::-1].index(marker)
    return run_id_from_run_dir(os.sep.join(parts[:cut]), runs_path)


def round_trip_check(master_rows_csv):
    """Verifica che `run_id_from_run_dir` riproduca il registro canonico riga per riga.

    Lista dei disaccordi, vuota se tutto torna. E' il test che la ricostruzione scritta a
    mano nel notebook post-hoc avrebbe fallito su tutte e 110 le righe.
    """
    import pandas as pd
    bad = []
    for _, r in pd.read_csv(master_rows_csv).iterrows():
        try:
            got = run_id_from_run_dir(r.run_dir)
        except ValueError as e:
            bad.append((str(r.run_dir), str(r.run_id), f"ERRORE: {e}"))
            continue
        if got != str(r.run_id):
            bad.append((str(r.run_dir), str(r.run_id), got))
    return bad
