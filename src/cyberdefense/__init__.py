"""Adaptive cyberattack detection and automated defense with GNNs and RL.

OpenMP rule (macOS): xgboost (Homebrew libomp) and torch (its own libomp) cannot both
run parallel code in one process. If torch loads first, XGBoost training segfaults; if
xgboost loads first, multithreaded torch ops segfault. So:

- baseline code (``cyberdefense.baselines``) imports xgboost first and uses torch only
  for TensorBoard logging;
- graph / GNN code (``cyberdefense.graph``) never imports xgboost or the baselines;
- a process that needs both (the test suite) imports xgboost first and runs torch on
  one thread (``tests/conftest.py``).
"""
