"""Experiment configuration as pydantic models.

Every script builds an ``ExperimentConfig`` so runs are reproducible and comparable.
Scripts enable CLI overrides with ``ExperimentConfig(_cli_parse_args=True)``, e.g.::

    uv run python scripts/train_baselines.py --baseline.models '["xgboost"]'
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Task = Literal["binary", "multiclass"]
Protocol = Literal["random", "unseen_attack", "unseen_host", "unseen_pair", "unseen_attack_host"]
BaselineModel = Literal["random_forest", "xgboost"]
GNNModel = Literal["host_prior", "mlp", "gcn", "sage", "gat"]
Ablation = Literal["none", "shuffled_structure", "topology_only", "no_propagation", "no_self_loops"]
ImbalanceStrategy = Literal["none", "class_weight", "undersample", "oversample", "smote"]


class PathsConfig(BaseModel):
    """Output locations shared by all experiments."""

    results_dir: Path = Path("results")
    tb_dir: Path = Path("results/tensorboard")


class TonIoTConfig(BaseModel):
    """Cleaning and preprocessing of the ToN-IoT network train/test CSV."""

    raw_csv: Path = Path("data/raw/ToN-IoT/train_test_network.csv")
    processed_dir: Path = Path("data/processed/ton_iot")
    eda_dir: Path = Path("results/eda")
    make_figures: bool = Field(True, description="write exploration figures during preparation")
    drop_duplicates: bool = Field(
        True, description="exact duplicate rows would otherwise land in both train and test"
    )
    text_cols: list[str] = Field(
        ["dns_query", "http_uri", "http_user_agent", "ssl_subject", "ssl_issuer"],
        description="free-text columns that act as signatures, not behaviour; dropped",
    )
    id_cols: list[str] = Field(
        ["src_ip", "src_port", "dst_ip", "dst_port"],
        description="kept for graph construction only, never used as model features",
    )
    log_transform: bool = Field(True, description="log1p numeric features (skewness 70-420)")
    min_category_freq: int = Field(
        20, ge=1, description="categories rarer than this in train collapse into 'infrequent'"
    )


class SplitConfig(BaseModel):
    """Stratified (on attack_type), group-aware (on feature_group) train/val/test split.

    The data is cut into ``n_folds`` stratified folds; ``test_folds`` of them form
    test, ``val_folds`` form val and the rest is train. Default 20/3/3 = 70/15/15.
    """

    n_folds: int = Field(20, ge=3)
    val_folds: int = Field(3, ge=1)
    test_folds: int = Field(3, ge=1)
    seed: int = 42
    max_group_size: int | None = Field(
        200,
        ge=1,
        description=(
            "identical-feature groups up to this size stay in one split; larger groups "
            "(e.g. 15k identical 1-SYN scanning flows) are split row-wise, otherwise "
            "stratification is impossible. None keeps every group intact."
        ),
    )

    @property
    def fractions(self) -> dict[str, float]:
        train = self.n_folds - self.val_folds - self.test_folds
        return {
            "train": train / self.n_folds,
            "val": self.val_folds / self.n_folds,
            "test": self.test_folds / self.n_folds,
        }


class RandomForestConfig(BaseModel):
    """Defaults reproduce the untuned baseline; tuning overrides them."""

    n_estimators: int = Field(300, gt=0)
    max_depth: int | None = Field(None, gt=0)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: Literal["sqrt", "log2"] | float = Field(
        "sqrt", description="features tried per split; a float is a fraction of all features"
    )
    max_samples: float | None = Field(
        None, gt=0, le=1, description="bootstrap sample size as a fraction; None = all rows"
    )
    criterion: Literal["gini", "entropy", "log_loss"] = "gini"
    class_weight_power: float = Field(
        1.0,
        ge=0,
        le=1,
        description=(
            "with imbalance=class_weight: weights = balanced ** power. 1 = fully balanced, "
            "0 = no weighting; in between trades rare-class recall against precision"
        ),
    )
    n_jobs: int = -1


class XGBoostConfig(BaseModel):
    """Defaults reproduce the untuned baseline; tuning overrides them."""

    n_estimators: int = Field(1000, gt=0, description="upper bound; early stopping on val")
    learning_rate: float = Field(0.1, gt=0)
    max_depth: int = Field(8, gt=0)
    min_child_weight: float = Field(1.0, ge=0)
    subsample: float = Field(0.8, gt=0, le=1)
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    gamma: float = Field(0.0, ge=0, description="minimum loss reduction to split")
    reg_lambda: float = Field(1.0, ge=0, description="L2 regularisation")
    reg_alpha: float = Field(0.0, ge=0, description="L1 regularisation")
    max_delta_step: float = Field(
        0.0, ge=0, description="caps each tree's update; helps logistic loss under imbalance"
    )
    class_weight_power: float = Field(1.0, ge=0, le=1, description="see RandomForestConfig")
    early_stopping_rounds: int = Field(50, gt=0)
    n_jobs: int = -1


class BaselineConfig(BaseModel):
    """Traditional ML baselines.

    The script runs every model x task x protocol x imbalance-strategy x seed combination.
    """

    models: list[BaselineModel] = ["random_forest", "xgboost"]
    tasks: list[Task] = ["binary", "multiclass"]
    protocols: list[Protocol] = Field(
        ["random", "unseen_attack", "unseen_host"],
        description=(
            "random: the stratified split as is. unseen_attack: for each attack type, "
            "remove it from train/val and measure how much of it is still detected in "
            "test. unseen_host: for each attacker host, remove every flow it sends or "
            "receives from train/val and measure detection of its test flows (an "
            "unfamiliar device). Both holdout protocols are binary only"
        ),
    )
    unseen_host_min_attack_flows: int = Field(
        500, gt=0, description="attacker hosts with at least this many attack flows are held out"
    )
    unseen_pair_folds: int = Field(
        5,
        ge=2,
        description=(
            "unseen_pair protocol: host pairs (undirected) are hashed into this many folds; "
            "each fold's pairs are removed from train/val and their test flows evaluated"
        ),
    )
    attack_host_min_test_flows: int = Field(
        100,
        gt=0,
        description=(
            "unseen_attack_host (Phase 3): an attack type is evaluated only if at least this "
            "many of its test flows remain and training keeps enough attack types and "
            "attacker hosts (the two settings below); infeasible types are skipped, never forced"
        ),
    )
    attack_host_min_train_attack_types: int = Field(
        3, ge=1, description="unseen_attack_host: attack types training must keep"
    )
    attack_host_min_train_attacker_hosts: int = Field(
        3, ge=1, description="unseen_attack_host: distinct attacker hosts training must keep"
    )
    imbalance: list[ImbalanceStrategy] = Field(
        ["class_weight"],
        description=(
            "none: plain fit. class_weight: balanced sample weights in the loss (the "
            "default; also used by the GNNs, so comparisons stay fair). undersample / "
            "oversample: random resampling of the training split. smote: synthetic "
            "minority rows (tabular ablation only; synthetic flows cannot enter a graph). "
            "Resampling touches the training split only, never val or test."
        ),
    )
    tune_threshold: bool = Field(
        True,
        description=(
            "binary only: pick the decision threshold that maximises F1 on val instead of "
            "0.5. Under imbalance the default threshold is rarely the best one."
        ),
    )
    fixed_fprs: list[float] = Field(
        [0.0025, 0.01],
        description=(
            "binary only: also report recall at these false-positive rates, with each "
            "threshold chosen on val normal traffic (0.0025 = 0.25%). Compares models at "
            "the same operating point instead of each one's own F1-tuned threshold"
        ),
    )
    seeds: list[int] = Field(
        [42],
        min_length=1,
        description="model seeds; several give mean +- std in summary_aggregated.csv",
    )
    use_tuned: bool = Field(
        False,
        description=(
            "load <tuning.out_dir>/<model>/best_params.json into the model configs; results "
            "go to results/baselines_tuned[_<suffix>] (suffix from the tuning folder name)"
        ),
    )
    random_forest: RandomForestConfig = RandomForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()


class ObjectiveWeights(BaseModel):
    """objective = macro_f1 * w_macro_f1 + unseen_recall * w_unseen - fpr * w_fpr."""

    macro_f1: float = Field(0.5, ge=0)
    unseen_recall: float = Field(0.5, ge=0)
    fpr: float = Field(
        2.0, ge=0, description="per unit of FPR; 2.0 makes +1% FPR cost 0.02 objective"
    )


class TuningConfig(BaseModel):
    """Hyperparameter search for the baselines, scored on the validation split only.

    Each trial fits one multiclass model (val macro-F1, which includes mitm) plus one
    binary model per tuning holdout (trained without that attack type; val recall on
    it and val FPR on normal traffic).
    """

    models: list[BaselineModel] = ["random_forest", "xgboost"]
    rf_trials: int = Field(30, gt=0, description="Random Forest: random search budget")
    xgb_trials: int = Field(40, gt=0, description="XGBoost: Optuna TPE budget")
    timeout_minutes: float | None = Field(
        None, gt=0, description="optional wall-clock cap per model; stops after this"
    )
    holdouts: list[str] = Field(
        ["xss", "dos", "ddos"],
        min_length=1,
        description=(
            "attack types whose unseen-attack recall is optimised during tuning. Their "
            "later unseen-attack test scores are 'tuned for' (optimistic); every other "
            "type stays genuinely unseen. The default keeps the hard targets mitm, "
            "injection and password out of tuning so their scores are genuine"
        ),
    )
    weights: ObjectiveWeights = ObjectiveWeights()
    prune: bool = Field(True, description="XGBoost: stop clearly bad trials early (median rule)")
    prune_after_holdouts: int = Field(
        2,
        ge=1,
        description=(
            "XGBoost: a trial can be pruned only after this many holdouts are scored. With 1, "
            "the first holdout alone decides, which pruned 28 of 40 trials in the first run"
        ),
    )
    train_fraction: float = Field(
        1.0, gt=0, le=1, description="stratified fraction of train used per trial (speed)"
    )
    out_dir: Path = Path("results/tuning_disjoint")


class ValidationConfig(BaseModel):
    """Dataset checks (scripts/validate_data.py)."""

    official_csv: Path | None = Field(
        None, description="official UNSW train_test_network.csv, if downloaded, for comparison"
    )
    out_dir: Path = Path("results/validation")


class FreezeConfig(BaseModel):
    """Snapshot of the final baselines (scripts/freeze_baselines.py)."""

    tag: str = Field("v1", pattern=r"^[A-Za-z0-9_.-]+$")
    out_dir: Path = Field(Path("frozen"), description="tracked in git, unlike results/")
    overwrite: bool = False


class GraphConfig(BaseModel):
    """Homogeneous host graph for the GNN stage (scripts/build_graph.py).

    Nodes are hosts (IP addresses), every flow is a directed edge src -> dst carrying the
    same preprocessed features as the baselines, and the task is edge classification.
    """

    node_features: Literal["constant", "degree"] = Field(
        "constant",
        description=(
            "constant: a single 1.0 per node, so all information comes from edge features "
            "and structure (no identity leak; E-GraphSAGE style). degree: log1p in/out flow "
            "counts and distinct neighbours, counted on message-passing edges only"
        ),
    )
    message_passing: Literal["train_only", "transductive"] = Field(
        "train_only",
        description=(
            "edges the GNN may propagate over. train_only: training edges only, for every "
            "phase (val/test edges are only classified). transductive: all edges' features "
            "(never labels) are visible, as in a deployment that sees all traffic"
        ),
    )
    protocol: Protocol = "random"
    holdout: str | None = Field(
        None, description="attack type (unseen_attack) or host IP (unseen_host) to hold out"
    )
    manifest: Path = Field(
        Path("frozen/baselines_v2/manifest.json"),
        description="frozen baseline manifest the splits must match (verify_splits)",
    )
    out_dir: Path = Path("data/processed/graph")
    report_dir: Path = Path("results/graph")
    make_figures: bool = True
    validate_holdouts: bool = Field(
        True,
        description=(
            "also build and validate the graph for every unseen_attack type and "
            "unseen_host host (not saved), proving each holdout leaves the training graph"
        ),
    )


class GNNConfig(BaseModel):
    """Edge-classification GNNs on the host graph (scripts/train_gnn.py).

    Every model shares one edge head (an MLP on ``[h_src, h_dst, edge_features]``) and
    one training recipe; only the node encoder differs:

    - ``mlp``: no graph, the head sees the edge features only (the graph-free control);
    - ``gcn`` / ``sage`` / ``gat``: GCNConv / SAGEConv (mean) / GATConv layers over the
      message-passing host graph.

    The graph comes from ``GraphConfig`` (node features, message passing); threshold
    tuning, fixed FPRs and the unseen-host list come from ``BaselineConfig``, so both
    stages are evaluated identically.
    """

    models: list[GNNModel] = Field(
        ["mlp", "gcn", "sage", "gat"],
        description=(
            "host_prior is the shortcut reference, not a neural net: a flow's score is its "
            "endpoints' attack history in the training labels (no flow features, no learning)"
        ),
    )
    tasks: list[Task] = ["binary", "multiclass"]
    protocols: list[Protocol] = ["random", "unseen_attack", "unseen_host"]
    ablation: Ablation = Field(
        "none",
        description=(
            "Phase 2 ablations (results go to results/gnn_<ablation>). shuffled_structure: "
            "convolution graph randomly rewired (degrees kept), node inputs intact. "
            "topology_only: no flow features anywhere; nodes get degree features, the head "
            "sees only the two endpoint embeddings. no_propagation: node inputs but zero "
            "message-passing layers. no_self_loops: self-loop flows removed from train, val "
            "and message passing (test unchanged; self-loop FPR is reported separately)"
        ),
    )
    holdouts: list[str] | None = Field(
        None,
        description="run only these attack types / hosts of the holdout protocols (None = all)",
    )
    seeds: list[int] = Field([42, 43, 44, 45, 46], min_length=1)
    hidden_dim: int = Field(128, gt=0)
    num_layers: int = Field(2, ge=1, description="message-passing layers (GNNs only)")
    dropout: float = Field(0.2, ge=0, lt=1)
    gat_heads: int = Field(4, ge=1, description="GAT attention heads; must divide hidden_dim")
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    batch_size: int = Field(4096, gt=0, description="training edges per gradient step")
    max_epochs: int = Field(100, gt=0)
    patience: int = Field(10, gt=0, description="early stopping: epochs without val-loss gain")
    min_delta: float = Field(1e-4, ge=0, description="smallest val-loss drop that counts")
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    resume: bool = Field(
        False,
        description="reuse finished runs' JSON in the output folder (after an interruption)",
    )
    compare_with: list[Path] = Field(
        [Path("frozen/baselines_v2/results/baselines/summary.csv")],
        description=(
            "baseline summary.csv files to compare against (scripts/compare_models.py). Add "
            "frozen/baselines_v2/results/baselines_tuned_disjoint/summary.csv for the tuned ones"
        ),
    )

    report_name: str | None = Field(
        None, description="compare_models output name (default: the GNN experiment name)"
    )

    @model_validator(mode="after")
    def _heads_divide_hidden(self) -> "GNNConfig":
        if self.hidden_dim % self.gat_heads:
            raise ValueError("gat_heads must divide hidden_dim")
        return self


class SurveyConfig(BaseModel):
    """Dataset suitability survey (scripts/survey_dataset.py, Phase 4, docs/08)."""

    dataset: Literal["ton_iot", "netflow_v3", "cic_ids2017", "unsw_nb15", "lanl"] = "ton_iot"
    name: str | None = Field(None, description="label in the report (default: dataset)")
    paths: list[Path] = Field(
        [],
        description=(
            "CSV files to read (ignored for ton_iot, which uses the prepared splits). "
            "lanl: the auth.txt(.gz) file(s); give the red-team file in lanl_redteam"
        ),
    )
    lanl_redteam: Path | None = None
    max_rows: int | None = Field(
        None, gt=0, description="read at most this many rows in total (large datasets)"
    )
    min_attack_flows_host: int = Field(
        100, gt=0, description="attacker hosts with at least this many attack flows"
    )
    out_dir: Path = Path("results/datasets")


class ExperimentConfig(BaseSettings):
    """Root config. Nested fields are overridable from the CLI as --section.field."""

    model_config = SettingsConfigDict(cli_implicit_flags=True, cli_prog_name="cyberdefense")

    name: str = "default"
    seed: int = 42
    paths: PathsConfig = PathsConfig()
    ton_iot: TonIoTConfig = TonIoTConfig()
    split: SplitConfig = SplitConfig()
    baseline: BaselineConfig = BaselineConfig()
    tuning: TuningConfig = TuningConfig()
    validation: ValidationConfig = ValidationConfig()
    freeze: FreezeConfig = FreezeConfig()
    graph: GraphConfig = GraphConfig()
    gnn: GNNConfig = GNNConfig()
    survey: SurveyConfig = SurveyConfig()
