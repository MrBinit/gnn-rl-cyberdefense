"""Traditional ML baselines (Random Forest, XGBoost).

xgboost is imported before anything pulls in torch (see ``cyberdefense/__init__``).
"""

import xgboost  # noqa: F401
