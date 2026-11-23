# 08 · Phase 4: dataset suitability for graph-based detection

**Status:** the survey tool is implemented, tested (8 tests) and calibrated on ToN-IoT.
- **Surveyed:**
  - ToN-IoT;
  - NF-UNSW-NB15-v3: **rejected** (§5a);
  - **NF-CSE-CIC-IDS2018-v3: conditionally suitable, best of the three** (§5b).
- **Not surveyed:** LANL, CIC-IDS2017.
- **Recommendation:** §5.

No model training, no graph redesign. Existing results are untouched.

## 1. Why ToN-IoT can't answer the research question (measured)
`scripts/survey_dataset.py --survey.name ToN-IoT` reproduces Phases 2–3:

| Diagnostic | ToN-IoT | Why it matters |
|---|---|---|
| hosts / sending hosts / host pairs | 776 / 51 / 1,004 | tiny relational structure; most hosts only receive |
| attacker hosts (≥ 100 attack flows) | 19 (9) | about 4 independent attacker groups in practice (docs/07) |
| **attack flows from mixed-role attackers** (hosts with ≥ 10% normal traffic) | **0.0%** | attackers only attack, so host identity *is* the label |
| source-host majority-label F1 (labels only, no learning) | **0.993** | host-reputation shortcut (docs/06) |
| host-pair majority F1 / test pairs seen in training | 0.999 / 99.8% | memorising connections solves the random split |
| unseen-attack + host: feasible types | 9 | but the cases are confounded: dst-reputation 5, novel-endpoint 2, clean 1 (scanning) |
| timestamps | **no** | no temporal graphs, no time-based splits |
| entity types | host, port, protocol | weak heterogeneous options |

A dataset that can show genuine GNN value needs the opposite profile:
- many hosts;
- attackers that also behave normally;
- weak host and pair shortcuts;
- several attack types from *different* hosts;
- timestamps;
- extra entity types.

## 2. What the survey measures (`cyberdefense.data.survey`)
Each dataset is normalised by an adapter (`netflow_v3`, `cic_ids2017`, `unsw_nb15`, `lanl`, `ton_iot`) to the
columns `src_ip, dst_ip, label, attack_type[, timestamp, dst_port, protocol, user]`. Then `survey()` computes:
- **Structure:** hosts, senders, undirected pairs, flows per pair, top-5 pair share, connected components.
- **Attacks:** types, attacker hosts, source hosts per type, victims.
- **Mixed roles:** attacker hosts with at least 10% normal traffic, and the share of attack flows they send.
- **Shortcuts** (random 70/30 split, training labels only):
  - source-host, destination-host and pair majority-label F1;
  - test pairs already seen in training;
  - per attack type, the share of its flows sent by hosts that commit *other* attacks (the "unseen attack is not
    unseen" effect).
- **Generalisation feasibility:**
  - unseen-host candidates;
  - per type, the unseen-attack + unseen-host split (`data.protocols.attack_host_feasibility`, the Phase 3
    code), with its destination-reputation and novel-endpoint exposure.
- **Time:** span, each attack type's time window, and how much normal traffic overlaps those windows (whether a
  time split is meaningful).
- **Heterogeneous potential:** available entity types and their cardinality.

## 3. Candidates (published facts)
| Dataset | What's documented | Expected shortcut risk | Verify with the survey |
|---|---|---|---|
| **UNSW-NB15** (original CSVs, or NF-UNSW-NB15-v3) | IXIA PerfectStorm testbed; **45 unique IP addresses across 3 networks** ([Ring et al.][ring]); 9 attack types; per-flow start time (`Stime`) | **confirmed**: 44 hosts, 4 attackers, source-host F1 1.000 (§5a) | done: **rejected** |
| **CIC-IDS2017** (GeneratedLabelledFlows, with IPs) | 5 days, CICFlowMeter flows with timestamps; **documented labelling, flow-construction and feature errors** ([Engelen et al.][engelen], [Lanvin et al.][lanvin]); corrected versions exist ([CNS2022 code][cns]) | **high**: attacks are run from a small attacker setup; use the corrected labels | attacker hosts, mixed-role share (botnet/infiltration victims may also act normally) |
| **CSE-CIC-IDS2018** (via **NF-CSE-CIC-IDS2018-v3**) | Larger AWS testbed over 10 days; the NetFlow v3 version has IPs, ports and **per-flow start/end timestamps**, plus binary and multi-class labels ([UQ NetFlow v3][nfv3], [record][nf18]). The CIC labelling issues reportedly carry over ([Engelen et al.][engelen]) | **medium**: more hosts and days; attacker diversity unknown | everything; the main detection candidate |
| **NetFlow v3 family** (NF-UNSW-NB15, NF-CSE-CIC-IDS2018, NF-ToN-IoT, NF-BoT-IoT) | One shared 53-feature NetFlow schema with timestamps across datasets ([UQ][nfv3], [NF-UNSW-NB15-v3][nfu]) | n/a | enables **cross-dataset** tests (train on one, test on another), the strongest unseen-host/unseen-network test |
| **LANL Comprehensive Cyber-Security Events** | 58 days, **12,425 users, 17,684 computers, 62,974 processes**, about 1.65 billion events (auth, process, flows, DNS); red-team events labelled in the auth data; de-identified, CC0 ([LANL][lanl], [OSTI][osti]) | **low for host identity** (red-team activity uses real credentials on real machines, so compromised users and computers also act normally; to be confirmed by the survey). **But**: one campaign, very sparse labels | mixed-role share; but there's **no unseen-attack test** (a single attack type) |

Not shortlisted:
- **DARPA OpTC:** host-provenance graphs at terabyte scale; a different task and infrastructure.
- **KDD99 / NSL-KDD:** no hosts.
- **Datasets without IPs** (for example the CIC-IDS2017 "MachineLearningCSV" files): no graph can be built.

## 4. Decision rules (applied to each survey row)
A dataset is suitable for testing *relational* GNN value only if:
1. **Attack flows from mixed-role attackers ≥ 20%.** ToN-IoT has 0%.
2. **Source-host shortcut F1 ≤ 0.95.** If it's higher, any graph model can win by remembering hosts.
3. **At least 10 attacker hosts with ≥ 100 attack flows**, and **at least 3 feasible unseen-attack + host
   types** whose targets have `target_dst_is_known_victim < 0.5` and `normal_both_ends_unseen` close to
   `target_dst_unseen` (no destination or novel-endpoint shortcut).
4. **Timestamps present,** so temporal and inductive graphs and time-based splits are possible.
5. Ideally **more than one entity type** (service/port, user, process), for a heterogeneous GNN.

Rules 1–3 are what Phases 2–3 showed to be decisive. A dataset that fails them would repeat the ToN-IoT result
regardless of model.

## 5a. Measured: NF-UNSW-NB15-v3 fails every relational rule
`results/datasets/NF-UNSW-NB15-v3/survey.json`: 2,365,424 flows, 27 days of timestamps.

| Diagnostic | NF-UNSW-NB15-v3 | ToN-IoT | Rule |
|---|---|---|---|
| hosts / senders / host pairs | **44 / 40 / 157** | 776 / 51 / 1,004 | smaller than ToN-IoT |
| attacker hosts | **4**; every one sends **all 9** attack types | 19 | rule 3 ✘ |
| attack flows from mixed-role attackers | **0%** | 0% | rule 1 ✘ |
| source-host majority F1 (labels only) | **1.000** | 0.993 | rule 2 ✘ |
| destination-host majority F1 | 1.000 (only 10 victims) | n/a | |
| test pairs seen in training | 100% | 99.8% | |
| each type's flows from hosts that commit other attacks | **100% for all 9 types** | 95–100% | |
| feasible unseen-attack + host types | **0**: holding out any type removes all 4 attackers and leaves no attacks to train on | 9 (1 clean) | rule 3 ✘ |
| timestamps | yes (27 days) | no | rule 4 ✔ |

**Verdict: rejected.**
- Any model that remembers the 4 attacker IPs (or the 10 victim IPs) is perfect.
- An "unseen attack" is impossible to define without also removing every attacker.
- The rich features (53 NetFlow fields, 64,617 distinct ports) and the timestamps don't change that.
- It remains usable only as a **cross-dataset test set** for a model trained elsewhere on the same NetFlow v3
  schema. There, its unseen hosts and unseen network are a genuine generalisation test.

## 5b. Measured: NF-CSE-CIC-IDS2018-v3, conditionally suitable
`results/datasets/NF-CSE-CIC-IDS2018-v3/survey.json`: 20,115,529 flows, 16 days (14 Feb – 2 Mar 2018). The survey
reads 7 of the 55 columns, peaks at 9.2 GB and takes 3.3 min.

| Diagnostic | NF-CSE-CIC-IDS2018-v3 | ToN-IoT | NF-UNSW-NB15-v3 | Rule |
|---|---|---|---|---|
| hosts / host pairs / components | **205,801 / 1,127,070 / 1** | 776 / 1,004 / 4 | 44 / 157 / 5 | |
| attack types / attacker hosts (≥ 100 attack flows) | 14 / 1,158 (37) | 9 / 19 (9) | 9 / 4 (4) | rule 3 ✔ (count) |
| attack flows from mixed-role attackers | **15.3%** | 0% | 0% | rule 1 ✘ (< 20%), but see below |
| source-host majority F1 | **0.922** | 0.993 | 1.000 | rule 2 ✔ (≤ 0.95) |
| destination-host / pair majority F1 | 0.967 / 0.974 | n/a / 0.999 | 1.000 / 0.904 | still high |
| test pairs seen in training | 97.1% | 99.8% | 100% | |
| timestamps | ✔ 16 days | ✘ | ✔ | rule 4 ✔ |

**Where the mixed roles are** (share of each type's attack flows sent by hosts with ≥ 10% normal traffic):
- **Bot: 100%** (8 infected internal hosts);
- **Infiltration: 99.9%**;
- web brute force: 82%; XSS: 91%; SQL injection: 79%;
- **every DoS, DDoS and FTP/SSH brute-force type: 0%.** These come from dedicated external AWS attackers
  (18.x / 13.x) that send 100% attack traffic, the ToN-IoT pattern again.

**Unseen-attack + host exposure** (all 14 types feasible by count):
- **Destination reputation:**
  - DoS, DDoS, FTP/SSH brute force and SQL injection: 100% of target flows hit a known victim;
  - each DoS/brute-force type comes from **one** dedicated attacker host;
  - a different host per type, so "unseen attack" here equals "unseen attacker", but the victim server is shared.
- **Clean:** **Bot** (known victim 0%, unseen destination 0%, mixed-role senders) and **Infiltration** (0.3%
  known victim; 26% unseen destination vs 0.4% for normal flows, so partly exposed to the novel-endpoint
  shortcut).
- **Novel endpoint:** XSS (92% unseen destination).

**Label quality:**
- Infiltration is attributed to **609 source hosts**, and web brute force to **561 hosts for only 1,618 flows**.
  That matches the time-window labelling problems documented for the CIC lineage ([Engelen et al.][engelen]).
- Treat these labels as noisy. Use corrected labels where available, or report those types separately.

**Time:** each day holds one attack family (14 Feb FTP/SSH brute force, 15–16 Feb DoS, 20–21 Feb DDoS, 22 Feb web
attacks, 28 Feb – 1 Mar infiltration, 2 Mar bot). Only 5% of normal flows fall inside each type's time window.
- A day-based split is automatically an unseen-attack split.
- Time of day is itself a shortcut, so time features must not be model inputs.

**Verdict: conditionally suitable.** It is far better than ToN-IoT and UNSW-NB15 (scale, connectivity,
timestamps, source-host F1 below 0.95), but it fails rule 1 overall. The relational question can only be
tested cleanly on **Bot** (and, with label caveats, Infiltration): internal compromised hosts that also behave
normally, which is exactly the scenario where graph structure should matter. For the external
DoS/DDoS/brute-force attacks, host and victim identity still decide.

## 5. Recommendation (after surveying three datasets)
1. **Adopt NF-CSE-CIC-IDS2018-v3 as the main detection dataset**, with a protocol that matches what it can
   support:
   - **Headline (relational) test: compromised internal hosts.**
     - Bot (and Infiltration, reported separately because of label noise).
     - Unseen-host split over the 8 bot hosts and the infiltration hosts.
     - These attackers also behave normally, so host identity is not the label.
   - **Day-based split.** Train on earlier days, test on later days. Each day is a new attack family, so this is
     a natural unseen-attack test. Exclude time features.
   - **Cross-dataset:** train on NF-CSE-CIC-IDS2018-v3, test on NF-UNSW-NB15-v3 (and NF-ToN-IoT-v3) with the
     same NetFlow schema. Every host is unseen.
   - **External DoS/DDoS/brute force:** report them, but label them **host/victim-confounded** (one dedicated
     attacker per type, known victim servers). No relational claim rests on them.
   - **Labels:** check web-attack and infiltration labels against corrected CIC labelling, or treat them as noisy.
2. **LANL** remains the best structural benchmark for lateral movement and the later defense stage
   (heterogeneous user–computer–process graphs). It's worth surveying if relational value can't be shown on
   CIC-IDS2018's bot/infiltration subset.
3. **Rejected:** NF-UNSW-NB15-v3 (measured, §5a; kept only as a cross-dataset test set). CIC-IDS2017 is optional
   (same lineage and labelling issues, smaller).

## 6. Run order
```bash
mkdir -p data/raw/nf data/raw/cic2017 data/raw/unsw data/raw/lanl
# Downloads (manual; check each licence page):
#  NetFlow v3 (NF-CSE-CIC-IDS2018-v3, NF-UNSW-NB15-v3): https://staff.itee.uq.edu.au/marius/NIDS_datasets/
#  CIC-IDS2017 GeneratedLabelledFlows (with IPs):        https://www.unb.ca/cic/datasets/ids-2017.html
#  UNSW-NB15 original CSVs (UNSW-NB15_1..4.csv):         https://research.unsw.edu.au/projects/unsw-nb15-dataset
#  LANL auth.txt.gz + redteam.txt.gz:                    https://csr.lanl.gov/data/cyber1/

uv run python scripts/survey_dataset.py --survey.name ToN-IoT                     # done (baseline row)
uv run python scripts/survey_dataset.py --survey.dataset netflow_v3 --survey.name NF-CSE-CIC-IDS2018-v3 \
  --survey.paths '["data/raw/nf/NF-CSE-CIC-IDS2018-v3.csv"]'
uv run python scripts/survey_dataset.py --survey.dataset netflow_v3 --survey.name NF-UNSW-NB15-v3 \
  --survey.paths '["data/raw/nf/NF-UNSW-NB15-v3.csv"]'
uv run python scripts/survey_dataset.py --survey.dataset cic_ids2017 --survey.name CIC-IDS2017 \
  --survey.paths "$(python3 -c 'import glob,json;print(json.dumps(sorted(glob.glob("data/raw/cic2017/*.csv"))))')"
uv run python scripts/survey_dataset.py --survey.dataset unsw_nb15 --survey.name UNSW-NB15 \
  --survey.paths '["data/raw/unsw/UNSW-NB15_1.csv","data/raw/unsw/UNSW-NB15_2.csv","data/raw/unsw/UNSW-NB15_3.csv","data/raw/unsw/UNSW-NB15_4.csv"]'
# LANL is about 1 billion auth events: survey a slice that contains red-team activity first
uv run python scripts/survey_dataset.py --survey.dataset lanl --survey.name LANL-auth-30M \
  --survey.max_rows 30000000 --survey.paths '["data/raw/lanl/auth.txt.gz"]' \
  --survey.lanl_redteam data/raw/lanl/redteam.txt.gz
```
→ `results/datasets/comparison.csv` (one row per dataset) and `results/datasets/<name>/survey.json`.

## 7. Proposed graph construction for the selected dataset
These are design principles. The graph is built only after the survey confirms the choice. Each one targets a
shortcut found on ToN-IoT.

**Homogeneous: temporal, inductive host graph (NF-CSE-CIC-IDS2018-v3)**
- **Time-windowed snapshots** (for example 5–15 minutes): nodes are the hosts active in the window, edges are its
  flows (edge classification, as now).
- **Window-local node features only:** degree, fan-out and port diversity computed *within the window*. No
  cross-time host profile, so a host can't carry its history (the Phase 2/3 shortcut) into the test period.
  Node indices are re-assigned per window, and there are no learnable IDs.
- **Splits:**
  - **time-based** (train on earlier days, test on later days);
  - **host-disjoint** (unseen host);
  - **type- and host-disjoint** (the Phase 3 protocol);
  - **cross-dataset** (train NF-CSE-CIC-IDS2018, test NF-UNSW-NB15 / NF-ToN-IoT).
- **Baselines get the same window-local host aggregates as features.** If RF plus those aggregates matches the
  GNN, the graph adds nothing (docs/07 §6.7).

**Heterogeneous**
- Node types: **host** and **service** (dst port + protocol).
- Edge types: host→host (flow, classified), host→service (uses) and service→host (offered by).
- Message passing then sees "which services a host talks to in this window", behaviour that transfers to unseen
  hosts better than host identity.
- For LANL: **user, computer, process** nodes, with authentication edges (user@src → dst, the classified edges)
  plus process-on-computer edges, in time windows.
- An alternative if edge-centric layers stay weak: a **flow-as-node graph** (flows linked when they share a host
  or service within a window).

**Leakage and shortcut guards carried over:**
- `verify_splits`-style frozen manifests;
- `audit_gnn` checks: message-passing masks, held-out hosts isolated, test-label invariance;
- the `host_prior` reference and the exposure table on every protocol;
- matched-actual-FPR evaluation with Wilson intervals (docs/07 §4).

## 8. Sources
[ring]: https://arxiv.org/pdf/1811.05372
[engelen]: https://lirias.kuleuven.be/retrieve/713070
[lanvin]: https://hal.archives-ouvertes.fr/hal-03775466
[cns]: https://www.github.com/GintsEngelen/CNS2022_Code
[nfv3]: https://staff.itee.uq.edu.au/marius/NIDS_datasets/
[nf18]: https://researchdata.edu.au/nf-cse-cic-ids2018-v3/3485181
[nfu]: https://researchdata.edu.au/nf-unsw-nb15-v3/3485172
[lanl]: https://csr.lanl.gov/data/cyber1/
[osti]: https://www.osti.gov/biblio/1179829

- Ring et al., *A survey of network-based intrusion detection data sets* (UNSW-NB15: 45 IPs, 3 networks):
  [arXiv 1811.05372][ring]
- Engelen, Rimmer, Joosen, *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*:
  [KU Leuven][engelen]; corrected labelling code: [CNS2022][cns]
- Lanvin et al., *Errors in the CICIDS2017 dataset…*: [HAL][lanvin]
- UQ NetFlow v3 datasets: [overview][nfv3], [NF-CSE-CIC-IDS2018-v3][nf18], [NF-UNSW-NB15-v3][nfu]
- Kent, *Comprehensive, Multi-Source Cyber-Security Events*, LANL 2015: [LANL][lanl], [OSTI][osti]
- UNSW-NB15 project page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
