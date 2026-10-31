"""Reproduce CUSUM-only simulation tuning and evaluation: python -m core.demo."""
import json
import numpy as np
from core.detectors import Cusum, cusum
from core.evaluate import run_kill_rule, score, tune
from core.model import Baseline, fit_baseline, fit_baselines, log_likelihood_ratio
from core.sim import MU, SCALE, NU, WARMUP, demo_strategy, simulate


def prepare(seed):
    strategies = simulate(300, seed, mu=MU, scale=SCALE, nu=NU)
    parameters = zip(*fit_baselines(strategies.returns[:, :WARMUP]))
    llr = np.array([
        log_likelihood_ratio(r, Baseline(*p))
        for r, p in zip(strategies.returns, parameters)
    ])
    return strategies, llr


def main():
    train, train_llr = prepare(1)
    test, test_llr = prepare(2)
    grid = np.round(np.linspace(0.5, 15.0, 59), 3)
    threshold, _ = tune(Cusum(), train.returns, train_llr, train.break_day, grid)
    metrics = score(run_kill_rule(Cusum(), test.returns, test_llr, [threshold]),
                    test.returns, test.break_day)
    returns = demo_strategy()
    base = fit_baseline(returns[:WARMUP])
    statistic = cusum(log_likelihood_ratio(returns[WARMUP:], base))
    hits = np.flatnonzero(statistic > threshold)
    result = {
        "threshold": threshold,
        "first_alarm_day_zero_based": int(hits[0] + WARMUP) if hits.size else None,
        "true_break_day_zero_based": 450,
        "test_mean_regret": float(metrics["mean_regret"][0]),
        "test_regret_standard_error": float(metrics["se_regret"][0]),
        "test_false_kills": int(metrics["false_kills"][0]),
        "test_missed_breaks": int(metrics["missed_breaks"][0]),
        "test_mean_delay_days": float(metrics["mean_delay"][0]),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
