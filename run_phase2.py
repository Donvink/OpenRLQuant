"""
run_phase2.py
─────────────
Phase 2 入口：PPO 训练。
直接代理到 train/train_ppo.py，保持四个 phase 入口统一。

Usage:
  python run_phase2.py --mode mlp --timesteps 500000
  python run_phase2.py --mode transformer --use-synthetic
  python run_phase2.py --mode curriculum
  python run_phase2.py --mode hyperopt --trials 30
"""
import sys
from train.train_ppo import main

if __name__ == "__main__":
    main()
