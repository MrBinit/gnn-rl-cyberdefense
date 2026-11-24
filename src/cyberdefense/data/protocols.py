"""Holdout definitions shared by every model (baselines, GNNs, RL).

Kept free of xgboost and torch so both the baselines and the GNN stage can import it
(see the OpenMP rule in ``cyberdefense/__init__``).

Protocols with a holdout:
- ``unseen_attack``: one attack type;
- ``unseen_host``: one attacker host, every flow it sends or receives;
- ``unseen_pair``: one fold of host pairs. Undirected pairs are hashed into folds, so a
  pair's flows in both directions leave training together, while both hosts keep their
  other connections. Holdout names are ``pair_fold_<k>_of_<n>``;
- ``unseen_attack_host`` (Phase 3): one attack type **and every host that sends it**, in
  any split. Every flow those hosts send or receive leaves training, so the type is new
  and so are its attackers. Holdout names are ``<type>@<host>,<host>,...``: the host set is
  computed once from all splits and carried in the name, so per-split masks agree.

``holdout_mask`` is what gets removed from train/val/message passing; ``target_mask`` is
what gets evaluated (identical except for unseen_attack_host, where the target is the
held-out type's flows plus the held-out hosts' own normal flows, not the held-out hosts'
attacks of *other*, seen types).
"""

import hashlib
import re

import numpy as np
import pandas as pd

from cyberdefense.config import ExperimentConfig, Protocol
from cyberdefense.data.schema import TARGET_BINARY, TARGET_MULTI

HOLDOUT_PROTOCOLS = ("unseen_attack", "unseen_host", "unseen_pair", "unseen_attack_host")
_PAIR_FOLD = re.compile(r"^pair_fold_(\d+)_of_(\d+)$")


def attacker_hosts(data: dict[str, pd.DataFrame], min_attack_flows: int) -> list[str]:
    """Source IPs with at least ``min_attack_flows`` attack flows (all splits), busiest first."""
    flows = pd.concat([df.loc[df[TARGET_BINARY] == 1, "src_ip"] for df in data.values()])
    counts = flows.value_counts()
    return counts[counts >= min_attack_flows].index.astype(str).tolist()


def pair_holdouts(n_folds: int) -> list[str]:
    return [f"pair_fold_{k}_of_{n_folds}" for k in range(n_folds)]


def undirected_pairs(df: pd.DataFrame) -> np.ndarray:
    """'a|b' with a < b for every row, so both directions map to the same pair."""
    src, dst = df["src_ip"].astype(str).to_numpy(), df["dst_ip"].astype(str).to_numpy()
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    return np.char.add(np.char.add(lo.astype(str), "|"), hi.astype(str))


def pair_fold(df: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Stable fold id per row from an MD5 hash of its undirected host pair."""
    pairs = undirected_pairs(df)
    uniq, inverse = np.unique(pairs, return_inverse=True)
    folds = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % n_folds for p in uniq])
    return folds[inverse]


def attack_host_holdout(data: dict[str, pd.DataFrame], attack_type: str) -> str:
    """``'<type>@<hosts>'``: the type plus every source host that sends it in any split."""
    flows = pd.concat([df.loc[df[TARGET_MULTI] == attack_type, "src_ip"] for df in data.values()])
    hosts = sorted(flows.astype(str).unique())
    if not hosts:
        raise ValueError(f"no flows of attack type {attack_type!r}")
    return f"{attack_type}@{','.join(hosts)}"


def parse_attack_host(holdout: str) -> tuple[str, list[str]]:
    attack_type, sep, hosts = str(holdout).partition("@")
    if not sep or not hosts:
        raise ValueError(f"unseen_attack_host holdout must look like <type>@<hosts>: {holdout!r}")
    return attack_type, hosts.split(",")


def holdout_hosts(protocol: Protocol, holdout: str | None) -> list[str]:
    """Hosts a holdout removes completely from training (empty for type/pair holdouts)."""
    if protocol == "unseen_host" and holdout:
        return [holdout]
    if protocol == "unseen_attack_host" and holdout:
        return parse_attack_host(holdout)[1]
    return []


def holdout_label(holdout: object) -> str:
    """Short display name: 'mitm@a,b,c' -> 'mitm (3 hosts)'; others unchanged."""
    text = str(holdout)
    if "@" not in text:
        return text
    attack_type, hosts = parse_attack_host(text)
    return f"{attack_type} ({len(hosts)} hosts)"


def _touches(df: pd.DataFrame, hosts: list[str]) -> np.ndarray:
    return (df["src_ip"].astype(str).isin(hosts) | df["dst_ip"].astype(str).isin(hosts)).to_numpy()


def holdout_mask(df: pd.DataFrame, protocol: Protocol, holdout: str) -> np.ndarray:
    """Rows removed from training: the held-out type, host(s) or host-pair fold."""
    if protocol == "unseen_attack":
        return (df[TARGET_MULTI] == holdout).to_numpy()
    if protocol == "unseen_host":
        return ((df["src_ip"] == holdout) | (df["dst_ip"] == holdout)).to_numpy()
    if protocol == "unseen_pair":
        match = _PAIR_FOLD.match(str(holdout))
        if not match:
            raise ValueError(
                f"unseen_pair holdout must look like pair_fold_<k>_of_<n>: {holdout!r}"
            )
        k, n = int(match[1]), int(match[2])
        return pair_fold(df, n) == k
    if protocol == "unseen_attack_host":
        attack_type, hosts = parse_attack_host(holdout)
        return _touches(df, hosts) | (df[TARGET_MULTI] == attack_type).to_numpy()
    raise ValueError(f"protocol {protocol!r} has no holdout")


def target_mask(df: pd.DataFrame, protocol: Protocol, holdout: str) -> np.ndarray:
    """Rows evaluated as the holdout (attack rows = the target, normal rows = its own traffic)."""
    if protocol != "unseen_attack_host":
        return holdout_mask(df, protocol, holdout)
    attack_type, hosts = parse_attack_host(holdout)
    is_type = (df[TARGET_MULTI] == attack_type).to_numpy()
    return is_type | (_touches(df, hosts) & (df[TARGET_BINARY] == 0).to_numpy())


def attack_host_feasibility(
    data: dict[str, pd.DataFrame], attack_type: str
) -> dict[str, float | int | str]:
    """What an unseen_attack_host split for ``attack_type`` leaves for training and test."""
    holdout = attack_host_holdout(data, attack_type)
    train, test = data["train"], data["test"]
    kept = train[~holdout_mask(train, "unseen_attack_host", holdout)]
    attacks = kept[kept[TARGET_BINARY] == 1]
    target = test[(test[TARGET_MULTI] == attack_type).to_numpy()]
    victims = set(attacks["dst_ip"].astype(str))
    return {
        "attack_type": attack_type,
        "holdout": holdout,
        "held_hosts": len(parse_attack_host(holdout)[1]),
        "train_rows_kept_share": len(kept) / max(len(train), 1),
        "train_attack_flows": len(attacks),
        "train_attack_types": int(attacks[TARGET_MULTI].nunique()),
        "train_attacker_hosts": int(attacks["src_ip"].nunique()),
        "train_normal_flows": int((kept[TARGET_BINARY] == 0).sum()),
        "test_target_flows": len(target),
        "target_dst_is_known_victim": float(target["dst_ip"].astype(str).isin(victims).mean())
        if len(target)
        else 0.0,
    }


def attack_host_holdouts(
    data: dict[str, pd.DataFrame],
    attack_types: list[str],
    min_test_flows: int,
    min_train_attack_types: int = 3,
    min_train_attacker_hosts: int = 3,
) -> list[str]:
    """Feasible unseen_attack_host holdouts; infeasible types are skipped, never forced."""
    keep = []
    for attack_type in attack_types:
        f = attack_host_feasibility(data, attack_type)
        if (
            f["test_target_flows"] >= min_test_flows
            and f["train_attack_types"] >= min_train_attack_types
            and f["train_attacker_hosts"] >= min_train_attacker_hosts
            and f["train_normal_flows"] > 0
        ):
            keep.append(str(f["holdout"]))
    return keep


def attack_host_holdouts_for(
    cfg: ExperimentConfig, data: dict[str, pd.DataFrame], attack_types: list[str]
) -> list[str]:
    """``attack_host_holdouts`` with the feasibility thresholds from ``cfg.baseline``."""
    b = cfg.baseline
    return attack_host_holdouts(
        data,
        attack_types,
        min_test_flows=b.attack_host_min_test_flows,
        min_train_attack_types=b.attack_host_min_train_attack_types,
        min_train_attacker_hosts=b.attack_host_min_train_attacker_hosts,
    )
