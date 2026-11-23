"""Graph construction and validation for the GNN stage.

Never import xgboost or ``cyberdefense.baselines`` from here (OpenMP rule in
``cyberdefense/__init__``). If xgboost is already loaded, torch is limited to one thread
so it cannot crash.
"""

import logging
import sys

import torch

if "xgboost" in sys.modules and torch.get_num_threads() > 1:
    logging.getLogger(__name__).warning(
        "xgboost is loaded in this process; running torch on 1 thread to avoid the "
        "OpenMP clash. Run graph/GNN code in a process without xgboost for full speed."
    )
    torch.set_num_threads(1)
