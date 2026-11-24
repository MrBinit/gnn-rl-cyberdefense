"""Dataset suitability survey for graph-based detection (Phase 4, docs/08).

Measures, for any candidate dataset, what Phases 2-3 found decisive (docs/06-07):

- graph size and diversity: hosts, senders, host pairs, flows per pair, components;
- attack structure: types, attacker hosts, how many hosts send each type;
- **mixed-role attackers**: do attacking hosts also send normal traffic? (On the first
  dataset, ToN-IoT, they never did, which turned host identity into a label.)
- **host-reputation shortcuts** (no learning, training labels only, random 70/30 split):
  source-host and host-pair majority-label F1, and how often a type's flows come from
  hosts that commit *other* attacks (what made "unseen attack" trivial);
- **generalisation feasibility**: unseen-host candidates and, per attack type, the
  unseen-attack + unseen-host split (``data.protocols.attack_host_feasibility``),
  including the destination-reputation and novel-endpoint exposure found in Phase 3;
- time: timestamps, span, whether attacks occupy separate time windows (temporal splits);
- heterogeneous potential: which entity types (ports/services, protocols, users,
  processes) are available besides hosts.

Every dataset is first normalised by an adapter (``prepared`` = our own splits) to the
columns ``src_ip``, ``dst_ip``, ``label`` (1 = attack), ``attack_type`` (``normal`` for
benign), optional ``timestamp`` (seconds) and optional entity columns (``dst_port``,
``protocol``, ``user``, ``process``).
Pandas only (no torch/xgboost); no model is trained.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from cyberdefense.data.protocols import attack_host_feasibility, parse_attack_host
from cyberdefense.data.schema import NORMAL_CLASS, TARGET_BINARY, TARGET_MULTI

ENTITY_COLS = ("dst_port", "protocol", "user", "process")
# The only NetFlow v3 columns the survey needs: reading just these keeps a 20M-row file
# (NF-CICIDS2018-v3, 3.9 GB) to a few GB of memory.
NETFLOW_COLS = [
    "FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "Label", "Attack",
]  # fmt: skip
CORE_COLS = ("src_ip", "dst_ip", TARGET_BINARY, TARGET_MULTI)


# ---------------------------------------------------------------------------- adapters
def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Common clean-up: string hosts, int label, lower-case types, benign -> 'normal'."""
    out = df.copy()
    out["src_ip"] = out["src_ip"].astype(str).str.strip()
    out["dst_ip"] = out["dst_ip"].astype(str).str.strip()
    out[TARGET_BINARY] = pd.to_numeric(out[TARGET_BINARY], errors="coerce").fillna(0).astype(int)
    types = out[TARGET_MULTI].astype(str).str.strip().str.lower()
    types = types.where(~types.isin(["benign", "normal", "", "nan", "-"]), NORMAL_CLASS)
    out[TARGET_MULTI] = np.where(out[TARGET_BINARY] == 0, NORMAL_CLASS, types)
    keep = [c for c in (*CORE_COLS, "timestamp", *ENTITY_COLS) if c in out]
    return out[keep].reset_index(drop=True)


def from_prepared(df: pd.DataFrame) -> pd.DataFrame:
    """Our own prepared splits (``data/processed/<dataset>``, NetFlow columns)."""
    return _finish(df.rename(columns={"PROTOCOL": "protocol"}))


def from_netflow_v3(df: pd.DataFrame) -> pd.DataFrame:
    """UQ NetFlow datasets (NF-UNSW-NB15, NF-CSE-CIC-IDS2018, NF-ToN-IoT, NF-BoT-IoT; v2/v3).

    v3 has ``FLOW_START_MILLISECONDS``; v2 has no timestamps.
    """
    out = pd.DataFrame(
        {
            "src_ip": df["IPV4_SRC_ADDR"],
            "dst_ip": df["IPV4_DST_ADDR"],
            TARGET_BINARY: df["Label"],
            TARGET_MULTI: df["Attack"],
            "dst_port": df["L4_DST_PORT"],
            "protocol": df["PROTOCOL"],
        }
    )
    if "FLOW_START_MILLISECONDS" in df:
        out["timestamp"] = pd.to_numeric(df["FLOW_START_MILLISECONDS"], errors="coerce") / 1000
    return _finish(out)


def from_cic_ids2017(df: pd.DataFrame) -> pd.DataFrame:
    """CICFlowMeter CSVs *with* IPs (CIC-IDS2017 "GeneratedLabelledFlows", CSE-CIC-IDS2018 Feb-20).

    Column names carry stray spaces in the original files; they are stripped first.
    """
    df = df.rename(columns=lambda c: str(c).strip())
    label = df["Label"].astype(str).str.strip()
    out = pd.DataFrame(
        {
            "src_ip": df["Source IP"] if "Source IP" in df else df["Src IP"],
            "dst_ip": df["Destination IP"] if "Destination IP" in df else df["Dst IP"],
            TARGET_BINARY: (label.str.upper() != "BENIGN").astype(int),
            TARGET_MULTI: label,
            "dst_port": df["Destination Port"] if "Destination Port" in df else df.get("Dst Port"),
            "protocol": df.get("Protocol"),
        }
    )
    if "Timestamp" in df:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        out["timestamp"] = (ts - pd.Timestamp(0)).dt.total_seconds()
    return _finish(out)


UNSW_COLS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
    "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]  # fmt: skip


def from_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    """Original UNSW-NB15_1..4.csv (no header, 49 columns, with IPs and Stime)."""
    if "srcip" not in df.columns:
        df = df.set_axis(UNSW_COLS[: df.shape[1]], axis=1)
    out = pd.DataFrame(
        {
            "src_ip": df["srcip"],
            "dst_ip": df["dstip"],
            TARGET_BINARY: df["Label"],
            TARGET_MULTI: df["attack_cat"].fillna("normal"),
            "dst_port": df["dsport"],
            "protocol": df["proto"],
            "timestamp": pd.to_numeric(df["Stime"], errors="coerce"),
        }
    )
    return _finish(out)


def from_lanl(auth: pd.DataFrame, redteam: pd.DataFrame) -> pd.DataFrame:
    """LANL "Comprehensive, Multi-Source Cyber-Security Events" authentication graph.

    ``auth``: time,src_user,dst_user,src_computer,dst_computer,auth_type,logon_type,
    auth_orientation,success. ``redteam``: time,user,src_computer,dst_computer. An auth
    event matching a red-team (time, user, src, dst) tuple is an attack; the red team is a
    single campaign, so there is one attack type ("redteam").
    """
    a = auth.iloc[:, :9].set_axis(
        [
            "time", "src_user", "dst_user", "src_computer", "dst_computer",
            "auth_type", "logon_type", "auth_orientation", "success",
        ],
        axis=1,
    )  # fmt: skip
    r = redteam.iloc[:, :4].set_axis(["time", "user", "src_computer", "dst_computer"], axis=1)
    key = ["time", "src_user", "src_computer", "dst_computer"]
    marks = r.rename(columns={"user": "src_user"})[key].astype(str).drop_duplicates()
    is_attack = (
        a[key].astype(str).merge(marks.assign(_hit=True), on=key, how="left")["_hit"].fillna(False)
    ).to_numpy(bool)
    out = pd.DataFrame(
        {
            "src_ip": a["src_computer"],
            "dst_ip": a["dst_computer"],
            TARGET_BINARY: is_attack.astype(int),
            TARGET_MULTI: np.where(is_attack, "redteam", NORMAL_CLASS),
            "timestamp": pd.to_numeric(a["time"], errors="coerce"),
            "user": a["src_user"],
            "protocol": a["auth_type"],
        }
    )
    return _finish(out)


ADAPTERS = {
    "prepared": from_prepared,
    "netflow_v3": from_netflow_v3,
    "cic_ids2017": from_cic_ids2017,
    "unsw_nb15": from_unsw_nb15,
}


def read_csvs(paths: Iterable[Path], max_rows: int | None = None, **kw) -> pd.DataFrame:
    """Concatenate CSV files, stopping after ``max_rows`` rows in total (None = all)."""
    frames, left = [], max_rows
    for p in paths:
        if left is not None and left <= 0:
            break
        frames.append(pd.read_csv(p, nrows=left, low_memory=False, **kw))
        if left is not None:
            left -= len(frames[-1])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------- survey
def _majority_f1(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> float:
    """F1 of predicting each test flow's label from its key's majority training label."""
    prior = train.groupby(keys)[TARGET_BINARY].mean()
    joined = test.join(prior.rename("p"), on=keys)
    pred = (joined["p"].fillna(train[TARGET_BINARY].mean()) >= 0.5).astype(int)
    y = test[TARGET_BINARY].to_numpy()
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _pair_codes(df: pd.DataFrame) -> np.ndarray:
    """Integer id per undirected host pair (memory-light ``undirected_pairs`` equivalent)."""
    codes, _ = pd.factorize(pd.concat([df["src_ip"], df["dst_ip"]], ignore_index=True))
    a, b = codes[: len(df)].astype(np.int64), codes[len(df) :].astype(np.int64)
    return np.minimum(a, b) * (codes.max() + 1) + np.maximum(a, b)


def _components(src: np.ndarray, dst: np.ndarray) -> list[int]:
    """Connected component sizes of the undirected host graph (union-find), largest first."""
    hosts, inv = np.unique(np.r_[src, dst], return_inverse=True)
    parent = np.arange(len(hosts))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(inv[: len(src)], inv[len(src) :], strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(hosts))])
    return sorted(np.bincount(roots)[np.unique(roots)].tolist(), reverse=True)


def survey(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    min_attack_flows_host: int = 100,
    mixed_role_min_normal_share: float = 0.1,
    attack_host_min_test_flows: int = 100,
) -> dict:
    """All suitability diagnostics for one normalised dataset (see module docstring)."""
    y = df[TARGET_BINARY].to_numpy()
    attacks = df[y == 1]
    out: dict = {"flows": len(df), "attack_share": float(y.mean())}

    # Graph size and diversity.
    src, dst = df["src_ip"].to_numpy(str), df["dst_ip"].to_numpy(str)
    pairs = pd.Series(_pair_codes(df), index=df.index)
    per_pair = pairs.value_counts()
    comps = _components(src, dst)
    out |= {
        "hosts": int(pd.unique(np.r_[src, dst]).size),
        "source_hosts": int(pd.unique(src).size),
        "undirected_pairs": int(per_pair.size),
        "flows_per_pair_median": float(per_pair.median()),
        "flows_per_pair_max": int(per_pair.max()),
        "top5_pairs_flow_share": float(per_pair.head(5).sum() / len(df)),
        "components": len(comps),
        "largest_component_hosts": comps[0],
    }

    # Attack structure and mixed-role attackers.
    types = attacks[TARGET_MULTI].value_counts()
    sent = df.groupby("src_ip")[TARGET_BINARY].agg(["size", "sum"])
    attackers = sent[sent["sum"] > 0]
    big = attackers[attackers["sum"] >= min_attack_flows_host]
    normal_share = 1 - attackers["sum"] / attackers["size"]
    mixed = attackers[normal_share >= mixed_role_min_normal_share]
    out |= {
        "attack_types": int(types.size),
        "attack_flows_per_type": types.to_dict(),
        "attacker_hosts": len(attackers),
        "attacker_hosts_with_min_attack_flows": len(big),
        "attackers_mixed_role": len(mixed),
        "attack_flows_from_mixed_role_hosts": float(
            attacks["src_ip"].isin(mixed.index).mean() if len(attacks) else 0.0
        ),
        "source_hosts_per_type": attacks.groupby(TARGET_MULTI)["src_ip"].nunique().to_dict(),
        "victim_hosts": int(attacks["dst_ip"].nunique()),
    }

    # Host-reputation shortcuts on a random 70/30 split (labels only, no learning).
    rng = np.random.default_rng(seed)
    in_train = rng.random(len(df)) < 0.7
    train, test = df[in_train], df[~in_train]
    pair_col = pairs.rename("pair")
    out |= {
        "shortcut_src_host_f1": _majority_f1(train, test, ["src_ip"]),
        "shortcut_dst_host_f1": _majority_f1(train, test, ["dst_ip"]),
        "shortcut_pair_f1": _majority_f1(
            train.assign(pair=pair_col[in_train].to_numpy()),
            test.assign(pair=pair_col[~in_train].to_numpy()),
            ["pair"],
        ),
        "test_pairs_seen_in_train": float(pair_col[~in_train].isin(set(pair_col[in_train])).mean()),
    }
    # "Unseen attack" exposure: a type's flows from hosts that commit other attack types.
    exposure = {}
    for t in types.index:
        hosts_t = attacks.loc[attacks[TARGET_MULTI] == t, "src_ip"]
        other = set(attacks.loc[attacks[TARGET_MULTI] != t, "src_ip"])
        exposure[t] = float(hosts_t.isin(other).mean())
    out["type_flows_from_hosts_with_other_attacks"] = exposure

    # Generalisation feasibility: unseen host, unseen attack + unseen host.
    splits = {"train": train, "test": test}
    feas = []
    for t in types.index:
        f = attack_host_feasibility(splits, t)
        held = parse_attack_host(f["holdout"])[1]
        kept = train[~(train["src_ip"].isin(held) | train["dst_ip"].isin(held))]
        seen = set(kept["src_ip"]) | set(kept["dst_ip"])
        target = test[test[TARGET_MULTI] == t]
        normal_test = test[test[TARGET_BINARY] == 0]
        feas.append(
            {
                **{k: v for k, v in f.items() if k != "holdout"},
                "target_dst_unseen": float((~target["dst_ip"].isin(seen)).mean())
                if len(target)
                else 0.0,
                "normal_both_ends_unseen": float(
                    (~normal_test["src_ip"].isin(seen) & ~normal_test["dst_ip"].isin(seen)).mean()
                )
                if len(normal_test)
                else 0.0,
                "feasible": bool(
                    f["test_target_flows"] >= attack_host_min_test_flows
                    and f["train_attack_types"] >= 3
                    and f["train_attacker_hosts"] >= 3
                ),
            }
        )
    out["attack_host_feasibility"] = feas
    out["unseen_attack_host_feasible_types"] = sum(f["feasible"] for f in feas)
    out["unseen_host_candidates"] = len(big)

    # Time.
    if "timestamp" in df and df["timestamp"].notna().any():
        ts = df["timestamp"]
        span = float(ts.max() - ts.min())
        windows = {}
        for t in types.index:
            tt = attacks.loc[attacks[TARGET_MULTI] == t, "timestamp"].dropna()
            windows[t] = [float(tt.min()), float(tt.max())] if len(tt) else None
        out |= {
            "has_timestamps": True,
            "time_span_hours": span / 3600,
            "attack_time_windows": windows,
            "normal_flows_during_attack_windows": float(
                np.mean(
                    [
                        ((ts[y == 0] >= w[0]) & (ts[y == 0] <= w[1])).mean()
                        for w in windows.values()
                        if w
                    ]
                )
            )
            if windows
            else 0.0,
        }
    else:
        out["has_timestamps"] = False

    # Heterogeneous potential.
    out["entity_types"] = ["host", *[c for c in ENTITY_COLS if c in df and df[c].notna().any()]]
    out["entity_cardinality"] = {c: int(df[c].nunique()) for c in ENTITY_COLS if c in df}
    return out


def headline(name: str, s: dict) -> dict:
    """One comparison-table row from a ``survey`` result."""
    return {
        "dataset": name,
        "flows": s["flows"],
        "hosts": s["hosts"],
        "source_hosts": s["source_hosts"],
        "host_pairs": s["undirected_pairs"],
        "attack_share": round(s["attack_share"], 3),
        "attack_types": s["attack_types"],
        "attacker_hosts": s["attacker_hosts"],
        "attackers_>=100_flows": s["attacker_hosts_with_min_attack_flows"],
        "mixed_role_attackers": s["attackers_mixed_role"],
        "attack_from_mixed_role": round(s["attack_flows_from_mixed_role_hosts"], 3),
        "src_host_shortcut_f1": round(s["shortcut_src_host_f1"], 3),
        "pair_shortcut_f1": round(s["shortcut_pair_f1"], 3),
        "test_pairs_seen": round(s["test_pairs_seen_in_train"], 3),
        "unseen_attack+host_types": s["unseen_attack_host_feasible_types"],
        "timestamps": s["has_timestamps"],
        "entity_types": ",".join(s["entity_types"]),
    }
