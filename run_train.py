"""
run_train.py
─────────────
第二步：PPO 强化学习训练

Input:   data/processed/   特征文件（由 run_data.py 生成）
         或自动生成合成数据（--use-synthetic）
Output:  models/           训练好的模型文件（.zip），供 run_trade.py 使用
         logs/             训练日志与实验记录

Usage:
  python run_train.py --mode mlp --timesteps 500000
  python run_train.py --mode transformer --use-synthetic
  python run_train.py --mode curriculum
  python run_train.py --mode hyperopt --trials 30

Next:
  python run_trade.py --mode alpaca --model-path models/ppo_mlp_final
"""
import sys
from train.train_ppo import main

if __name__ == "__main__":
    main()
