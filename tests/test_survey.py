"""Phase 4 dataset survey: adapters normalise every format; diagnostics match known answers."""

import numpy as np
import pandas as pd
import pytest

from cyberdefense.data.survey import (
    from_cic_ids2017,
    from_lanl,
    from_netflow_v3,
    from_ton_iot,
    from_unsw_nb15,
    headline,
    survey,
)


def crafted(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Host 'a1' only attacks (dedicated); host 'm1' is mixed-role (half normal)."""
    rng = np.random.default_rng(seed)
    victims = [f"v{i}" for i in range(20)]
    rows = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            rows.append(("a1", rng.choice(victims), 1, "ddos", i))
        elif kind == 1:
            rows.append(("m1", rng.choice(victims), 1, "scan", i))
        elif kind == 2:
            rows.append(("m1", rng.choice(victims), 0, "normal", i))
        else:
            rows.append((f"n{i % 30}", rng.choice(victims), 0, "normal", i))
    return pd.DataFrame(rows, columns=["src_ip", "dst_ip", "label", "attack_type", "timestamp"])


def test_survey_measures_reputation_and_mixed_roles() -> None:
    s = survey(crafted(), min_attack_flows_host=100)
    assert s["attacker_hosts"] == 2
    assert s["attackers_mixed_role"] == 1  # m1 sends 50% normal
    assert s["attack_flows_from_mixed_role_hosts"] == pytest.approx(0.5, abs=0.01)
    # a1's flows are perfectly predicted by host; m1's are a coin flip -> F1 well below 1.
    assert 0.5 < s["shortcut_src_host_f1"] < 0.95
    assert s["type_flows_from_hosts_with_other_attacks"] == {"ddos": 0.0, "scan": 0.0}
    assert s["has_timestamps"]
    assert s["entity_types"] == ["host"]
    row = headline("crafted", s)
    assert row["mixed_role_attackers"] == 1
    assert row["attack_types"] == 2


def test_dedicated_attackers_make_the_host_shortcut_perfect() -> None:
    df = crafted()
    df = df[df["src_ip"] != "m1"]
    s = survey(df)
    assert s["shortcut_src_host_f1"] == pytest.approx(1.0)
    assert s["attackers_mixed_role"] == 0


def test_feasibility_flags_too_few_types() -> None:
    s = survey(crafted())
    feas = {f["attack_type"]: f for f in s["attack_host_feasibility"]}
    assert set(feas) == {"ddos", "scan"}
    assert not any(f["feasible"] for f in feas.values())  # < 3 types remain in training
    assert s["unseen_attack_host_feasible_types"] == 0


def test_netflow_v3_adapter() -> None:
    raw = pd.DataFrame(
        {
            "IPV4_SRC_ADDR": ["1.1.1.1", "2.2.2.2"],
            "IPV4_DST_ADDR": ["3.3.3.3", "3.3.3.3"],
            "L4_DST_PORT": [80, 53],
            "PROTOCOL": [6, 17],
            "FLOW_START_MILLISECONDS": [1000, 2000],
            "Label": [0, 1],
            "Attack": ["Benign", "DoS"],
        }
    )
    df = from_netflow_v3(raw)
    assert df["attack_type"].tolist() == ["normal", "dos"]
    assert df["timestamp"].tolist() == [1.0, 2.0]
    assert {"dst_port", "protocol"} <= set(df.columns)


def test_cic_adapter_strips_column_spaces_and_parses_time() -> None:
    raw = pd.DataFrame(
        {
            " Source IP": ["1.1.1.1", "2.2.2.2"],
            " Destination IP": ["3.3.3.3", "3.3.3.3"],
            " Destination Port": [80, 22],
            " Protocol": [6, 6],
            " Timestamp": ["3/7/2017 8:55", "3/7/2017 9:00"],
            " Label": ["BENIGN", "SSH-Patator"],
        }
    )
    df = from_cic_ids2017(raw)
    assert df["label"].tolist() == [0, 1]
    assert df["attack_type"].tolist() == ["normal", "ssh-patator"]
    assert df["timestamp"].iloc[1] - df["timestamp"].iloc[0] == 300


def test_unsw_adapter_reads_headerless_rows() -> None:
    row = ["59.166.0.1", 1, "149.171.126.6", 53, "udp"] + [0] * 43 + [0]
    row[28] = 1421927414  # Stime
    row[47] = np.nan  # attack_cat empty for normal traffic
    attack = list(row)
    attack[0], attack[47], attack[48] = "175.45.176.1", "Exploits", 1
    df = from_unsw_nb15(pd.DataFrame([row, attack]))
    assert df["attack_type"].tolist() == ["normal", "exploits"]
    assert df["timestamp"].iloc[0] == 1421927414


def test_lanl_adapter_marks_red_team_events() -> None:
    auth = pd.DataFrame(
        [
            [1, "U1@D", "U1@D", "C1", "C2", "NTLM", "Network", "LogOn", "Success"],
            [2, "U66@D", "U66@D", "C17", "C3", "NTLM", "Network", "LogOn", "Success"],
        ]
    )
    red = pd.DataFrame([[2, "U66@D", "C17", "C3"]])
    df = from_lanl(auth, red)
    assert df["label"].tolist() == [0, 1]
    assert df["attack_type"].tolist() == ["normal", "redteam"]
    assert set(df.columns) >= {"user", "timestamp"}


def test_ton_iot_adapter(tmp_cfg) -> None:
    from cyberdefense.data.pipeline import build_dataset
    from cyberdefense.data.ton_iot import SPLITS, load_split

    cfg = tmp_cfg.model_copy(
        update={"ton_iot": tmp_cfg.ton_iot.model_copy(update={"make_figures": False})}
    )
    build_dataset(cfg)
    df = from_ton_iot(pd.concat([load_split(cfg.ton_iot, s) for s in SPLITS]))
    s = survey(df, min_attack_flows_host=1)
    assert s["flows"] == len(df)
    assert not s["has_timestamps"]
