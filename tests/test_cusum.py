import numpy as np
from scipy import stats
from core.detectors import Cusum, cusum
from core.evaluate import run_kill_rule
from core.model import Baseline, log_likelihood_ratio


def test_worked_example_and_strict_threshold():
    np.testing.assert_allclose(cusum([1, -0.5, -1, 2]), [1, 0.5, 0, 2])
    assert not Cusum().fires(2, 2)
    assert Cusum().fires(2.01, 2)


def test_recurrence_matches_maximum_suffix_log_likelihood():
    llr = np.random.default_rng(42).normal(size=60)
    expected = [max(0, max(sum(llr[k:t+1]) for k in range(t+1)))
                for t in range(len(llr))]
    np.testing.assert_allclose(cusum(llr), expected, atol=1e-12)
    np.testing.assert_allclose(cusum(np.stack([llr, -llr])),
                               np.stack([cusum(llr), cusum(-llr)]))


def test_likelihood_ratio_and_closed_form():
    base = Baseline(0.0015, 0.008, 4)
    r = np.array([-0.03, -0.0015, 0, 0.0015, 0.03])
    expected = stats.t.logpdf(r, base.nu, -base.mu, base.sigma) - stats.t.logpdf(
        r, base.nu, base.mu, base.sigma)
    closed = (base.nu + 1) / 2 * (
        np.log1p((r - base.mu)**2 / (base.nu * base.sigma**2))
        - np.log1p((r + base.mu)**2 / (base.nu * base.sigma**2)))
    np.testing.assert_allclose(log_likelihood_ratio(r, base), expected, atol=1e-12)
    np.testing.assert_allclose(closed, expected, atol=1e-12)
    assert expected[1] > 0 and expected[3] < 0


def test_no_change_models_are_indistinguishable_at_zero_mean():
    np.testing.assert_array_equal(log_likelihood_ratio([-1, 0, 1], Baseline(0, 1, 4)), 0)


def test_alarm_day_is_traded_then_sixty_days_off_and_reset():
    returns = np.ones((1, 67))
    llr = np.ones_like(returns)
    run = run_kill_rule(Cusum(), returns, llr, [1], start=0, redeploy=60)
    np.testing.assert_array_equal(np.flatnonzero(run.fired[0, 0]), [1, 63])
    np.testing.assert_array_equal(np.flatnonzero(run.size[0, 0]), [0, 1, 62, 63])


def test_empty_time_axis():
    assert cusum(np.empty((3, 0))).shape == (3, 0)
