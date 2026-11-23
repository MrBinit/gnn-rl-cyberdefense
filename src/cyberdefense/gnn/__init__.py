"""Edge-classification GNNs (MLP control, GCN, GraphSAGE, GAT) on the host graph.

Same OpenMP rule as ``cyberdefense.graph``: never import xgboost or
``cyberdefense.baselines`` from here. Importing ``cyberdefense.graph`` applies its guard.
"""

import cyberdefense.graph  # noqa: F401  (limits torch to 1 thread if xgboost is loaded)
