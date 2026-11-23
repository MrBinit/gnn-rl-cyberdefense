"""Survey a network-intrusion dataset's suitability for graph-based detection (Phase 4).

No model training; reads the data and writes diagnostics (docs/08). Usage::

    uv run python scripts/survey_dataset.py                                  # ToN-IoT
    uv run python scripts/survey_dataset.py --survey.dataset netflow_v3 \\
        --survey.name NF-UNSW-NB15-v3 --survey.paths '["data/raw/nf/NF-UNSW-NB15-v3.csv"]'
    uv run python scripts/survey_dataset.py --survey.dataset cic_ids2017 \\
        --survey.name CIC-IDS2017 --survey.paths '["data/raw/cic2017/Monday.csv", "..."]'
    uv run python scripts/survey_dataset.py --survey.dataset lanl --survey.max_rows 20000000 \\
        --survey.paths '["data/raw/lanl/auth.txt.gz"]' \\
        --survey.lanl_redteam data/raw/lanl/redteam.txt.gz

Writes results/datasets/<name>/survey.json and appends/updates the row in
results/datasets/comparison.csv (one row per surveyed dataset).
"""

import json
import logging

import pandas as pd

from cyberdefense.config import ExperimentConfig
from cyberdefense.data.survey import (
    ADAPTERS,
    NETFLOW_COLS,
    from_lanl,
    headline,
    read_csvs,
    survey,
)
from cyberdefense.data.ton_iot import SPLITS, load_split
from cyberdefense.utils import setup_logging

log = logging.getLogger("survey_dataset")


def main() -> None:
    setup_logging()
    cfg = ExperimentConfig(_cli_parse_args=True)
    sc = cfg.survey
    name = sc.name or sc.dataset
    if sc.dataset != "ton_iot":
        files = [*sc.paths, *([sc.lanl_redteam] if sc.lanl_redteam else [])]
        missing = [str(p) for p in files if not p.exists()]
        if not sc.paths or missing:
            raise SystemExit(
                f"{name}: no data to survey - missing {missing or 'all files (empty path list)'}. "
                "Download the dataset first (links in docs/08 §6) and put the CSVs there."
            )
    if sc.dataset == "ton_iot":
        df = ADAPTERS["ton_iot"](pd.concat([load_split(cfg.ton_iot, s) for s in SPLITS]))
    elif sc.dataset == "lanl":
        if sc.lanl_redteam is None:
            raise SystemExit("lanl needs --survey.lanl_redteam")
        auth = read_csvs(sc.paths, sc.max_rows, header=None)
        df = from_lanl(auth, pd.read_csv(sc.lanl_redteam, header=None))
    else:
        kw = {"header": None} if sc.dataset == "unsw_nb15" else {}
        if sc.dataset == "netflow_v3":
            kw["usecols"] = lambda c: c in NETFLOW_COLS
        df = ADAPTERS[sc.dataset](read_csvs(sc.paths, sc.max_rows, **kw))
    log.info("%s: %s flows loaded", name, f"{len(df):,}")

    result = survey(df, seed=cfg.seed, min_attack_flows_host=sc.min_attack_flows_host)
    out = sc.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "survey.json").write_text(json.dumps(result, indent=2, default=str))
    row = pd.DataFrame([headline(name, result)])
    table_path = sc.out_dir / "comparison.csv"
    if table_path.exists():
        old = pd.read_csv(table_path)
        row = pd.concat([old[old["dataset"] != name], row], ignore_index=True)
    row.to_csv(table_path, index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        log.info("headline:\n%s", pd.DataFrame([headline(name, result)]).T.to_string())
        feas = pd.DataFrame(result["attack_host_feasibility"])
        log.info("unseen attack + host feasibility:\n%s", feas.to_string(index=False))
    log.info("survey -> %s ; comparison table -> %s", out / "survey.json", table_path)


if __name__ == "__main__":
    main()
