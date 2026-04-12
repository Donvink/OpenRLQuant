"""
train/train_ppo.py  — Phase 2 PPO Training (full replacement)
"""
import argparse, json, logging, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CONFIG, DATA_DIR, LOG_DIR, MODEL_DIR
from environment.trading_env import TradingEnv, make_env
from environment.backtester import BuyAndHoldBenchmark, compute_metrics, compare_strategies, print_comparison_table
from utils.helpers import set_seed, setup_logging
from utils.experiment_tracker import ExperimentTracker

logger = logging.getLogger("train_ppo")


# ── Synthetic data builder (no internet needed) ────────────────────────────────

def build_synthetic_feature_store(symbols, n_days=1200, seed=42):
    from features.feature_engineer import TechnicalFeatures, MacroFeatures, RollingNormalizer
    np.random.seed(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)

    def make_stock(i, base=100.0):
        np.random.seed(i)
        r = np.random.normal(0.0002 + i * 0.00005, 0.018, n_days)
        c = base * np.cumprod(1 + r)
        return pd.DataFrame({
            "open": c * (1 + np.random.normal(0, 0.004, n_days)),
            "high": c * (1 + abs(np.random.normal(0, 0.007, n_days))),
            "low":  c * (1 - abs(np.random.normal(0, 0.007, n_days))),
            "close": c, "volume": np.random.lognormal(16, 0.6, n_days),
        }, index=dates).rename_axis("date")

    bases = [150, 280, 2800, 200, 140, 160, 90, 300, 120, 250, 180, 320, 95, 400, 220]
    raw = {s: make_stock(i, bases[i % len(bases)]) for i, s in enumerate(symbols)}
    raw["SPY"] = make_stock(99, 380)
    vix = pd.Series(np.clip(np.random.lognormal(3.0, 0.4, n_days), 10, 60), index=dates, name="vix")

    tech = TechnicalFeatures()
    macro_df = MacroFeatures().build(raw, vix)
    norm = RollingNormalizer(252)
    store = {}
    for s in symbols:
        df = tech.compute(raw[s].copy())
        df = pd.concat([df, macro_df.reindex(df.index).ffill()], axis=1)
        df = norm.fit_transform(df, exclude_cols=["open","high","low","close","volume"])
        df = df.dropna(how="all")
        if len(df) >= 200:
            store[s] = df
    logger.info(f"Synthetic store: {len(store)} symbols x {n_days} days")
    return store, vix


def load_feature_store(symbols, start, end):
    cache = str(DATA_DIR / "processed")
    from features.feature_engineer import FeaturePipeline
    p = FeaturePipeline()
    try:
        store = p.load(cache, symbols)
        if store:
            logger.info(f"Loaded {len(store)} symbols from cache")
            return store, None
    except Exception:
        pass
    from data.market_data import MarketDataLoader
    loader = MarketDataLoader(polygon_key=os.getenv("POLYGON_API_KEY",""),
                              cache_dir=DATA_DIR/"cache", use_cache=True)
    market_data = loader.get_ohlcv_universe(symbols, start, end)
    vix = loader.get_vix(start, end)
    store = p.build(market_data, vix=vix)
    p.save(store, cache)
    return store, vix


# ── Vec env builder ────────────────────────────────────────────────────────────

def build_vec_env(feature_store, symbols, n_envs=4, mode="train", env_kwargs=None):
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    env_kwargs = env_kwargs or {}
    def _make(seed):
        def _init():
            env = TradingEnv(feature_store=feature_store, symbols=symbols, mode=mode, **env_kwargs)
            env.reset(seed=seed)
            return env
        return _init
    factories = [_make(i * 100) for i in range(n_envs)]
    try:
        vec = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)
    except Exception:
        vec = DummyVecEnv(factories)
    return vec


# ── MLP training ───────────────────────────────────────────────────────────────

def train_mlp(feature_store, symbols, total_timesteps=300_000, n_envs=2,
              save_dir="models", experiment_name="ppo_mlp"):
    from stable_baselines3 import PPO
    from train.callbacks import TradingEvalCallback, CheckpointCallback, MetricsLoggerCallback, EarlyStoppingCallback

    logger.info("="*60 + "\nTRAINING: PPO + MLP Policy\n" + "="*60)

    env_kwargs = dict(initial_capital=1_000_000, lookback_window=30,
                      episode_length=126, reward_type="log_return", # sharpe is too noisy for short episodes
                      drawdown_penalty=0.1, turnover_penalty=0.005,
                      transaction_cost_bps=5.0)

    train_env = build_vec_env(feature_store, symbols, n_envs, "train", env_kwargs)
    val_env = TradingEnv(feature_store=feature_store, symbols=symbols, mode="val", **env_kwargs)

    model = PPO("MlpPolicy", train_env,
                learning_rate=3e-4, n_steps=max(1024//n_envs, 128),
                batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs={"net_arch": dict(pi=[128,64], vf=[128,64])},
                tensorboard_log=str(LOG_DIR/"tensorboard"), verbose=1)

    callbacks = [
        TradingEvalCallback(val_env,
                            eval_freq=max(2_000//n_envs, 500),
                            n_eval_episodes=5,
                            save_path=str(Path(save_dir)/"checkpoints"),
                            best_model_name=f"{experiment_name}_best"),
        CheckpointCallback(save_freq=max(20_000//n_envs, 2000),
                        save_path=str(Path(save_dir)/"checkpoints"),
                        name_prefix=experiment_name),
        MetricsLoggerCallback(log_freq=500),
        EarlyStoppingCallback(
            monitor="eval/mean_sharpe",
            patience=8,
            min_delta=0.01,
        ),
    ]

    tracker = ExperimentTracker(experiment_name)
    with tracker.start_run(experiment_name):
        tracker.log_params({"algo":"PPO","policy":"MLP","timesteps":total_timesteps,
                            "n_envs":n_envs,"n_stocks":len(symbols)})
        t0 = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    tb_log_name=experiment_name, reset_num_timesteps=True)
        tracker.log_metrics({"training_min": (time.time()-t0)/60})

    final = Path(save_dir) / f"{experiment_name}_final"
    model.save(final)
    logger.info(f"Model saved: {final}.zip")
    train_env.close()
    return model


# ── Transformer training ───────────────────────────────────────────────────────

def train_transformer(feature_store, symbols, total_timesteps=1_000_000, n_envs=2,
                      d_model=64, n_heads=4, n_layers=2, save_dir="models",
                      experiment_name="ppo_transformer"):
    from stable_baselines3 import PPO
    from train.policy_network import TransformerFeaturesExtractor, make_transformer_policy_kwargs
    from train.callbacks import TradingEvalCallback, CheckpointCallback, EarlyStoppingCallback, MetricsLoggerCallback

    logger.info("="*60 + f"\nTRAINING: PPO + Transformer (d={d_model}, h={n_heads}, L={n_layers})\n" + "="*60)

    lookback = 30
    env_kwargs = dict(initial_capital=1_000_000, lookback_window=lookback,
                      episode_length=126, reward_type="log_return",
                      drawdown_penalty=0.1, turnover_penalty=0.005, transaction_cost_bps=5.0)

    train_env = build_vec_env(feature_store, symbols, n_envs, "train", env_kwargs)
    val_env   = TradingEnv(feature_store=feature_store, symbols=symbols, mode="val", **env_kwargs)

    # Probe dims from single env
    probe = TradingEnv(feature_store=feature_store, symbols=symbols, **env_kwargs)
    n_features = probe.n_features
    port_dim   = probe.n + 5

    policy_kwargs = make_transformer_policy_kwargs(
        n_stocks=len(symbols), n_features=n_features, lookback=lookback,
        portfolio_state_dim=port_dim, d_model=d_model,
        n_heads=n_heads, n_transformer_layers=n_layers, use_transformer=True)

    model = PPO("MlpPolicy", train_env,
                learning_rate=1e-4, n_steps=max(2048//n_envs,128),
                batch_size=128, n_epochs=8, gamma=0.99, gae_lambda=0.95,
                clip_range=0.15, ent_coef=0.002, vf_coef=0.5, max_grad_norm=0.3,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(LOG_DIR/"tensorboard"), verbose=1)

    n_params = sum(p.numel() for p in model.policy.parameters())
    logger.info(f"Policy parameters: {n_params:,}")

    callbacks = [
        TradingEvalCallback(val_env,
                            eval_freq=max(5_000//n_envs, 1000),
                            n_eval_episodes=5,
                            save_path=str(Path(save_dir)/"checkpoints"),
                            best_model_name=f"{experiment_name}_best"),
        CheckpointCallback(save_freq=max(50_000//n_envs, 5000),
                            save_path=str(Path(save_dir)/"checkpoints"),
                            name_prefix=experiment_name),
        MetricsLoggerCallback(log_freq=1000),
        EarlyStoppingCallback(patience=8, min_delta=0.02),
    ]

    tracker = ExperimentTracker(experiment_name)
    with tracker.start_run(experiment_name):
        tracker.log_params({"algo":"PPO","policy":"Transformer","d_model":d_model,
                            "n_heads":n_heads,"n_layers":n_layers,
                            "n_params":n_params,"timesteps":total_timesteps})
        model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=experiment_name)

    final = Path(save_dir) / f"{experiment_name}_final"
    model.save(final)
    train_env.close()
    return model


# ── Curriculum training ────────────────────────────────────────────────────────

def train_curriculum(feature_store, symbols, vix=None, save_dir="models"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from train.curriculum import CurriculumManager, CURRICULUM_STAGES
    from train.callbacks import TradingEvalCallback

    logger.info("="*60 + "\nTRAINING: Curriculum Learning\n" + "="*60)
    curriculum = CurriculumManager(CURRICULUM_STAGES, feature_store, symbols, vix)
    model = None

    while not curriculum.is_complete():
        stage = curriculum.current_stage
        stage_syms = curriculum.select_symbols(stage)
        logger.info(f"\n▶ {stage.name} — {stage.description}")

        env_kw = dict(initial_capital=1_000_000, lookback_window=stage.lookback_window,
                      episode_length=stage.episode_length, reward_type=stage.reward_type,
                      transaction_cost_bps=stage.transaction_cost_bps)

        train_env = DummyVecEnv([make_env(feature_store, stage_syms, mode="train", seed=i, **env_kw) for i in range(2)])
        val_env   = TradingEnv(feature_store=feature_store, symbols=stage_syms, mode="val", **env_kw)

        if model is None:
            model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=512, batch_size=64,
                        policy_kwargs={"net_arch": dict(pi=[256,128], vf=[256,128])},
                        tensorboard_log=str(LOG_DIR/"tensorboard"), verbose=0)
        else:
            model.set_env(train_env)

        eval_cb = TradingEvalCallback(val_env, eval_freq=3_000, n_eval_episodes=5,
                                      save_path=str(Path(save_dir)/"curriculum"),
                                      best_model_name=f"stage{curriculum.current_stage_idx+1}_best",
                                      curriculum_manager=curriculum)

        model.learn(total_timesteps=stage.timesteps, callback=eval_cb,
                    tb_log_name=f"curriculum_s{curriculum.current_stage_idx+1}",
                    reset_num_timesteps=False)
        train_env.close()

        # Force eval if curriculum didn't auto-advance
        if not curriculum.is_complete():
            sharpes = []
            for ep in range(8):
                obs, _ = val_env.reset(seed=ep)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, t, tr, info = val_env.step(action)
                    done = t or tr
                sharpes.append(info.get("sharpe_ratio", 0.0))
            curriculum.record_eval(float(np.mean(sharpes)))

    logger.info(curriculum.summary())
    model.save(Path(save_dir) / "curriculum_final")
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, feature_store, symbols, n_eval_episodes=10, save_dir="models"):
    logger.info("\n" + "="*60 + "\nEVALUATION: Test Set\n" + "="*60)
    env_kw = dict(initial_capital=1_000_000, lookback_window=30, episode_length=63,
                  reward_type="log_return", drawdown_penalty=0.1, turnover_penalty=0.005,
                  transaction_cost_bps=5.0)
    test_env = TradingEnv(feature_store=feature_store, symbols=symbols, mode="test", **env_kw)

    reports, all_pv = [], []
    for ep in range(n_eval_episodes):
        obs, _ = test_env.reset(seed=ep * 13 + 42)
        pv = [test_env.portfolio.total_value]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, t, tr, info = test_env.step(action)
            pv.append(info["total_value"])
            done = t or tr
        r = compute_metrics(pd.Series(pv, dtype=float))
        reports.append(r)
        all_pv.append(pv)
        logger.info(f"  Ep {ep+1:2d}: Return={r.total_return:+.2%} Sharpe={r.sharpe_ratio:.3f} DD={r.max_drawdown:.2%}")

    summary = {
        "mean_sharpe": round(float(np.mean([r.sharpe_ratio for r in reports])), 4),
        "std_sharpe":  round(float(np.std([r.sharpe_ratio  for r in reports])), 4),
        "mean_return": round(float(np.mean([r.total_return  for r in reports])), 4),
        "mean_max_dd": round(float(np.mean([r.max_drawdown  for r in reports])), 4),
        "pct_profitable": round(float(np.mean([r.total_return > 0 for r in reports])), 3),
    }
    logger.info(f"\nSummary: {summary}")

    with open(Path(save_dir) / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlp","transformer","curriculum","hyperopt"], default="mlp")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--n-stocks", type=int, default=5)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--save-dir", default=str(MODEL_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging("INFO", LOG_DIR / "train_ppo.log")
    set_seed(args.seed)

    default_syms = ["AAPL","MSFT","GOOGL","NVDA","META","JPM","JNJ","XOM","HD","WMT"]
    symbols = args.symbols or default_syms[:args.n_stocks]

    if args.use_synthetic:
        feature_store, vix = build_synthetic_feature_store(symbols, seed=args.seed)
    else:
        feature_store, vix = load_feature_store(symbols, args.start, args.end)

    valid = [s for s in symbols if s in feature_store]
    logger.info(f"Valid symbols: {valid}")
    if len(valid) < 2:
        logger.error("Need >= 2 symbols. Use --use-synthetic")
        return

    if args.mode == "mlp":
        model = train_mlp(feature_store, valid, args.timesteps, args.n_envs, args.save_dir)
    elif args.mode == "transformer":
        model = train_transformer(feature_store, valid, args.timesteps, args.n_envs,
                                  args.d_model, args.n_heads, args.n_layers, args.save_dir)
    elif args.mode == "curriculum":
        model = train_curriculum(feature_store, valid, vix, args.save_dir)
    elif args.mode == "hyperopt":
        from train.hyperopt import HyperparamOptimizer
        opt = HyperparamOptimizer(feature_store, valid, n_trials=args.trials,
                                  short_train_steps=min(args.timesteps//5, 30_000),
                                  save_dir=str(Path(args.save_dir)/"hyperopt"))
        best = opt.run()
        logger.info(f"Best params: {best}")
        model = train_mlp(feature_store, valid, args.timesteps, args.n_envs, args.save_dir, "ppo_optimized")

    summary = evaluate_model(model, feature_store, valid, save_dir=args.save_dir)
    logger.info(f"\n✓ Phase 2 complete | Sharpe={summary['mean_sharpe']:.3f} | Return={summary['mean_return']:+.2%}")


if __name__ == "__main__":
    main()
