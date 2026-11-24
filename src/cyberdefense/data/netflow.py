"""NetFlow v3 datasets (UQ): schema, loading, cleaning, window features and sampling.

Default source: **NF-CICIDS2018-v3** (20,115,529 flows, 53 NetFlow fields, 14 attack
types). Every decision below comes from the full-data exploration in docs/01; any NetFlow
v3 file with the same columns (NF-UNSW-NB15-v3, NF-ToN-IoT-v3...) loads the same way.

Feature policy:
- **never features**: IPs and ports (host identity), ``FLOW_START/END_MILLISECONDS`` (one
  attack family per day: time is a label shortcut), ``DNS_QUERY_ID`` (random id);
- **TTL** is off by default: each attack tool has one fixed TTL, so it fingerprints the
  machine (``use_ttl_features``);
- **removed as useless** (docs/01 §7): ``FTP_COMMAND_RET_CODE`` (constant),
  ``LONGEST_FLOW_PKT`` (= ``MAX_IP_PKT_LEN``), ``RETRANSMITTED_IN/OUT_PKTS`` (r >= 0.998
  with the byte counts), ``ICMP_TYPE`` (type*256+code; ``ICMP_IPV4_TYPE`` kept);
- protocol/flag/type *codes* are categorical, everything else numeric.

Window features (docs/01 §8): per 5-minute window and computed on the **complete**
deduplicated traffic *before* any sampling, so sampling can't erase them: flows of the
same host pair, the sender's flows / distinct destinations / distinct destination ports,
the receiver's flows / distinct senders, and whether each endpoint is an internal
(private-range) address. They are behaviour and address-space facts, not identities, and
always stored; they enter the models only with ``use_window_features`` (the GNN builds the
same structure itself, so the fair baseline is RF/XGBoost *with* them).

Sampling (docs/01 §9.1): ``windows`` (default) keeps **whole time windows**, so graph
neighbourhoods stay intact: every window holding a reliably labelled attack plus a seeded
random share of attack-free windows, with attack flows capped per (window, type) to tame
the DDoS bursts. ``per_class`` caps rows per class at random (tabular-only experiments).
"""

import ipaddress
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals

from cyberdefense.config import DatasetConfig
from cyberdefense.data.schema import (
    DAY,
    FEATURE_GROUP,
    NOISY_LABEL,
    NORMAL_CLASS,
    TARGET_BINARY,
    TARGET_MULTI,
    TIMESTAMP,
    WINDOW,
)

RAW_LABEL, RAW_ATTACK = "Label", "Attack"
RAW_IDS = {
    "IPV4_SRC_ADDR": "src_ip",
    "L4_SRC_PORT": "src_port",
    "IPV4_DST_ADDR": "dst_ip",
    "L4_DST_PORT": "dst_port",
}
RAW_TIME = ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")
DROPPED = ("DNS_QUERY_ID",)
TTL_COLS = ["MIN_TTL", "MAX_TTL"]
# Removed after the exploration (docs/01 §7), with the reason recorded in the metadata.
REMOVED_FEATURES = {
    "FTP_COMMAND_RET_CODE": "constant (cardinality 1)",
    "LONGEST_FLOW_PKT": "identical to MAX_IP_PKT_LEN (Spearman r = 1.0)",
    "RETRANSMITTED_IN_PKTS": "duplicates RETRANSMITTED_IN_BYTES (r = 0.998)",
    "RETRANSMITTED_OUT_PKTS": "duplicates RETRANSMITTED_OUT_BYTES (r = 1.0)",
    "ICMP_TYPE": "type*256+code; ICMP_IPV4_TYPE kept",
}

NUMERIC_COLS = [
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_OUT_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "DNS_TTL_ANSWER",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]  # fmt: skip
# Codes, not quantities (a DNS query type of 28 is not "more" than 1).
CATEGORICAL_COLS = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE",
]  # fmt: skip
WINDOW_FEATURES = [
    "WIN_PAIR_FLOWS", "WIN_SRC_FLOWS", "WIN_SRC_DSTS", "WIN_SRC_DST_PORTS",
    "WIN_DST_FLOWS", "WIN_DST_SRCS", "SRC_INTERNAL", "DST_INTERNAL",
]  # fmt: skip
RAW_COLUMNS = {
    *RAW_IDS, *RAW_TIME, *DROPPED, *TTL_COLS, *REMOVED_FEATURES,
    *NUMERIC_COLS, *CATEGORICAL_COLS, RAW_LABEL, RAW_ATTACK,
}  # fmt: skip

# Attack-type counts of the published NF-CICIDS2018-v3 file (surveyed, docs/08 §5b).
PUBLISHED_COUNTS = {
    "normal": 17_514_626,
    "ddos_attack-hoic": 1_032_311,
    "ftp-bruteforce": 386_720,
    "ddos_attacks-loic-http": 288_589,
    "bot": 207_703,
    "ssh-bruteforce": 188_474,
    "infilteration": 188_152,
    "dos_attacks-slowhttptest": 105_550,
    "dos_attacks-hulk": 100_076,
    "dos_attacks-goldeneye": 61_300,
    "dos_attacks-slowloris": 36_040,
    "ddos_attack-loic-udp": 3_450,
    "brute_force_-web": 1_618,
    "brute_force_-xss": 480,
    "sql_injection": 440,
}


def numeric_cols(cfg: DatasetConfig) -> list[str]:
    return (
        NUMERIC_COLS
        + (TTL_COLS if cfg.use_ttl_features else [])
        + (WINDOW_FEATURES if cfg.use_window_features else [])
    )


def feature_cols(cfg: DatasetConfig) -> list[str]:
    return numeric_cols(cfg) + CATEGORICAL_COLS


def normalise_attack(values: pd.Series) -> pd.Series:
    """'DDOS_attack-HOIC' -> 'ddos_attack-hoic'; 'Benign' -> 'normal'."""
    names = values.astype(str).str.strip().str.lower()
    return names.where(~names.isin(["benign", "normal"]), NORMAL_CLASS)


@dataclass(frozen=True)
class CleanReport:
    raw_rows: int
    excluded_day_rows: int
    duplicate_rows: int
    deduplicated_rows: int
    clean_rows: int
    sampling: str
    windows_total: int
    windows_kept: int
    attack_windows_kept: int
    rows_per_class_before_sampling: dict[str, int]
    rows_per_class_after_sampling: dict[str, int]
    removed_features: dict[str, str]
    never_feature_cols: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """Read the CSV in chunks with compact dtypes (float32 numbers, categorical text)."""
    float_cols = [*NUMERIC_COLS, *TTL_COLS, *DROPPED, *REMOVED_FEATURES]
    dtypes = {c: "float32" for c in float_cols}
    dtypes |= {c: "category" for c in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", RAW_ATTACK]}
    dtypes |= {c: "float64" for c in RAW_TIME}  # epoch milliseconds need float64
    chunks = pd.read_csv(
        cfg.raw_csv,
        dtype=dtypes,
        chunksize=cfg.chunk_rows,
        nrows=cfg.raw_max_rows,
        low_memory=False,
    )
    frames = list(chunks)
    if len(frames) == 1:
        return frames[0]
    # Chunks have different category sets; concat would silently fall back to object
    # strings (gigabytes on 20M rows), so merge the categories explicitly.
    cats = [c for c, t in frames[0].dtypes.items() if isinstance(t, pd.CategoricalDtype)]
    merged = {c: union_categoricals([f[c] for f in frames]) for c in cats}
    out = pd.concat([f.drop(columns=cats) for f in frames], ignore_index=True)
    for c in cats:
        out[c] = merged[c]
    return out[frames[0].columns]


def _host_codes(src: pd.Series, dst: pd.Series) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Shared integer code per IP address across source and destination columns."""
    s_cat, d_cat = src.astype("category"), dst.astype("category")
    hosts = s_cat.cat.categories.union(d_cat.cat.categories)
    s = hosts.get_indexer(s_cat.cat.categories)[s_cat.cat.codes.to_numpy()]
    d = hosts.get_indexer(d_cat.cat.categories)[d_cat.cat.codes.to_numpy()]
    return s.astype(np.int64), d.astype(np.int64), hosts


def _is_internal(hosts: pd.Index) -> np.ndarray:
    """Private / link-local / unspecified address ranges (the monitored network side)."""
    out = np.zeros(len(hosts), dtype=bool)
    for i, h in enumerate(hosts.astype(str)):
        try:
            out[i] = ipaddress.ip_address(h).is_private
        except ValueError:
            out[i] = False
    return out


def window_features(
    src: np.ndarray, dst: np.ndarray, dst_port: np.ndarray, window: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-flow structure counts within its time window (on whatever rows are passed)."""
    k = pd.DataFrame({"w": window, "s": src, "d": dst, "p": dst_port})

    def size(keys: list[str]) -> np.ndarray:
        return k.groupby(keys, sort=False)["w"].transform("size").to_numpy()

    def distinct(keys: list[str], col: str) -> np.ndarray:
        counts = k[[*keys, col]].drop_duplicates().groupby(keys, sort=False).size()
        return k.join(counts.rename("_n"), on=keys)["_n"].to_numpy()

    return {
        "WIN_PAIR_FLOWS": size(["w", "s", "d"]),
        "WIN_SRC_FLOWS": size(["w", "s"]),
        "WIN_SRC_DSTS": distinct(["w", "s"], "d"),
        "WIN_SRC_DST_PORTS": distinct(["w", "s"], "p"),
        "WIN_DST_FLOWS": size(["w", "d"]),
        "WIN_DST_SRCS": distinct(["w", "d"], "s"),
    }


def _window_sample(
    types: np.ndarray, window: np.ndarray, cfg: DatasetConfig
) -> tuple[np.ndarray, int, int, int]:
    """Positions to keep under ``windows`` sampling; also (#windows, #kept, #attack kept)."""
    rng = np.random.default_rng(cfg.sample_seed)
    clean_attack = (types != NORMAL_CLASS) & ~np.isin(types, cfg.noisy_attack_types)
    all_windows = np.unique(window)
    attack_windows = np.unique(window[clean_attack])
    quiet = np.setdiff1d(all_windows, attack_windows)
    n_quiet = min(len(quiet), round(cfg.normal_window_ratio * len(attack_windows)))
    chosen = np.union1d(attack_windows, rng.choice(quiet, size=n_quiet, replace=False))
    keep = np.isin(window, chosen)

    cap = cfg.max_attack_flows_per_window
    if cap is not None:  # thin the DDoS bursts; normal traffic stays whole
        idx = np.flatnonzero(keep & (types != NORMAL_CLASS))
        groups = pd.Series(idx).groupby([window[idx], types[idx]], sort=False).indices
        for members in groups.values():
            if len(members) > cap:
                drop = rng.choice(members, size=len(members) - cap, replace=False)
                keep[idx[drop]] = False
    return np.flatnonzero(keep), len(all_windows), len(chosen), len(attack_windows)


def _class_sample(types: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """``per_class``: at most ``max_normal_rows`` normal and ``max_rows_per_attack`` per type."""
    rng = np.random.default_rng(cfg.sample_seed)
    keep = []
    for cls, idx in pd.Series(types).groupby(types, sort=True).indices.items():
        cap = cfg.max_normal_rows if cls == NORMAL_CLASS else cfg.max_rows_per_attack
        if cap is not None and len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def clean(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, CleanReport]:
    """Return the prepared table (ids, time, day, window, features, targets) and a report."""
    if missing := RAW_COLUMNS - set(df.columns):
        raise ValueError(f"NetFlow v3 CSV is missing columns: {sorted(missing)}")
    if unexpected := set(df.columns) - RAW_COLUMNS:
        raise ValueError(f"unhandled columns in NetFlow v3 CSV: {sorted(unexpected)}")
    raw_rows = len(df)

    # 1. Drop excluded days (27 Feb: 828 flows), then exact duplicates.
    start = df[RAW_TIME[0]].to_numpy(np.float64)
    day = pd.to_datetime(start, unit="ms").strftime("%Y-%m-%d").to_numpy()
    in_day = ~np.isin(day, cfg.exclude_days)
    # Deduplicate on what the prepared table keeps: rows differing only in dropped columns
    # (DNS_QUERY_ID, end time, removed features) would otherwise become exact copies.
    kept_cols = [
        *RAW_IDS, RAW_TIME[0], *NUMERIC_COLS, *(TTL_COLS if cfg.use_ttl_features else []),
        *CATEGORICAL_COLS, RAW_LABEL, RAW_ATTACK,
    ]  # fmt: skip
    dup = pd.util.hash_pandas_object(df[kept_cols], index=False).duplicated().to_numpy()
    duplicate_rows = int((dup & in_day).sum())
    kept = np.flatnonzero(in_day & ~dup if cfg.drop_duplicates else in_day)
    del dup
    # Work on positions and single columns: copying the full 20M-row frame costs ~5 GB.
    start, day = start[kept], day[kept]
    deduplicated = len(kept)

    # 2. Window features on the complete traffic, before any sampling.
    window = (start // (cfg.window_seconds * 1000)).astype(np.int64)
    src, dst, hosts = _host_codes(df["IPV4_SRC_ADDR"].iloc[kept], df["IPV4_DST_ADDR"].iloc[kept])
    internal = _is_internal(hosts)
    win = window_features(src, dst, df["L4_DST_PORT"].to_numpy()[kept], window)
    win["SRC_INTERNAL"], win["DST_INTERNAL"] = internal[src], internal[dst]
    del src, dst

    # 3. Sample positions on labels and windows alone; only kept rows are converted.
    types = normalise_attack(df[RAW_ATTACK].iloc[kept]).to_numpy()
    before = pd.Series(types).value_counts().to_dict()
    if cfg.sampling == "windows":
        pos, n_windows, n_kept, n_attack = _window_sample(types, window, cfg)
    else:
        pos = _class_sample(types, cfg)
        n_windows = n_kept = len(np.unique(window))
        n_attack = len(np.unique(window[types != NORMAL_CLASS]))
    df = df.iloc[kept[pos]]

    out = pd.DataFrame(index=df.index)
    for raw, name in RAW_IDS.items():
        out[name] = df[raw].astype(str) if "ADDR" in raw else df[raw].astype("int32")
    out[TIMESTAMP] = start[pos] / 1000
    out[DAY] = day[pos]
    out[WINDOW] = window[pos]
    for col in [*NUMERIC_COLS, *(TTL_COLS if cfg.use_ttl_features else [])]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
    for col, values in win.items():
        out[col] = values[pos].astype("float32")
    for col in CATEGORICAL_COLS:
        # Integer codes as text categories ('6' not '6.0'); L7_PROTO keeps its decimals.
        values = df[col]
        if col != "L7_PROTO":
            values = pd.to_numeric(values, errors="coerce").fillna(-1).astype("int64")
        out[col] = values.astype(str).astype("category")
    out[TARGET_MULTI] = types[pos]
    out[TARGET_BINARY] = (out[TARGET_MULTI] != NORMAL_CLASS).astype("int8")
    out = out.reset_index(drop=True)
    out[TARGET_MULTI] = out[TARGET_MULTI].astype("category")
    out[NOISY_LABEL] = out[TARGET_MULTI].isin(cfg.noisy_attack_types).astype("int8")
    # Flows with identical features (ignoring ids and time) share a group; splits use it.
    out[FEATURE_GROUP] = pd.util.hash_pandas_object(out[feature_cols(cfg)], index=False).astype(
        "int64"
    )

    report = CleanReport(
        raw_rows=raw_rows,
        excluded_day_rows=int((~in_day).sum()),
        duplicate_rows=duplicate_rows,
        deduplicated_rows=deduplicated,
        clean_rows=len(out),
        sampling=cfg.sampling,
        windows_total=n_windows,
        windows_kept=n_kept,
        attack_windows_kept=n_attack,
        rows_per_class_before_sampling={k: int(v) for k, v in before.items()},
        rows_per_class_after_sampling={
            k: int(v) for k, v in out[TARGET_MULTI].value_counts().items()
        },
        removed_features=REMOVED_FEATURES,
        never_feature_cols=[
            *RAW_IDS,
            *RAW_TIME,
            *DROPPED,
            *([] if cfg.use_ttl_features else TTL_COLS),
        ],
    )
    return out, report
