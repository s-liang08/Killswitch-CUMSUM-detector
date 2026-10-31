# Killswitch - CUMSUM detector

**Original team project:** [Killswitch](https://github.com/leilei67/Killswitch)

A focused extraction and mathematical walkthrough of the CUSUM detector in
[Killswitch](https://github.com/leilei67/Killswitch), a team project that asks
whether a trading strategy has broken or is experiencing a bad run.

This repository documents part of our collaborative work. **Other members of
the team are Leiyi Sun and Zijian Wang.** The original implementation was
developed as a team project.

The detector accumulates evidence for a change from a healthy positive return
mean to a negative mean. It combines a Student-t return model with a one-sided
log-likelihood-ratio CUSUM.

## Run

With Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run pytest -q
uv run python -m core.demo
```

Alternatively, in a Python virtual environment:

```bash
python -m pip install -e . pytest
python -m pytest -q
python -m core.demo
```

The demo tunes on 300 simulated strategies (seed 1), evaluates on 300 independent
strategies (seed 2), then reports the first alarm on the original demo strategy.
It uses synthetic data and needs no downloaded market data.

Verified with the committed lockfile: **17 tests passed**. The demo reproduced
CUSUM's upstream results:

| Output | Value |
|---|---:|
| Training-selected threshold | 4.0 |
| First demo alarm (zero-based day) | 550 |
| True demo break (zero-based day) | 450 |
| Test mean regret | 15.8583% of starting capital |
| Standard error of test mean regret | 0.9143 percentage points |
| Test false kills | 93 / 300 strategies |
| Test missed breaks | 28 / 200 broken strategies |
| Mean delay among detected test breaks | 68.1163 days |

The demo path's delay (100 days) differs from the average delay across the test
set.

A minimal example, after installing dependencies:

```python
from core.detectors import Cusum, cusum

s = cusum([1.0, -0.5, -1.0, 2.0])
print(s)                         # [1.  0.5 0.  2. ]
print(Cusum().fires(s, 2.0))      # all False: the comparison is strictly >
```

Here the inputs are **log-likelihood ratios**. This complete example converts
simulated daily returns into the detector statistic:

```python
from core.model import fit_baseline, log_likelihood_ratio
from core.detectors import cusum
from core.sim import demo_strategy

returns = demo_strategy()  # Daily decimal returns: 1% = 0.01.
base = fit_baseline(returns[:250])
s = cusum(log_likelihood_ratio(returns[250:], base))
print(s[:5])
```

Fit the baseline on the initial period only. The detector starts from zero on
the first monitored day, after the 250-day warm-up.

## 1. Healthy and broken return models

Let $r_t$ be a daily return. Under the healthy model,

```math
H_0:\quad r_t=\mu+\sigma Z_t,\qquad Z_t\sim t_\nu.
```

Under the broken model,

```math
H_1:\quad r_t=\mu_b+\sigma Z_t,\qquad \mu_b=-k|\mu|,\quad k=1.
```

The two models share scale $\sigma$ and degrees of freedom $\nu$; only location
changes. The healthy baseline has $\mu>0$, so the default alternative has
$\mu_b=-\mu$.

The Student-t density with location $m$ is

```math
f(r;m,\sigma,\nu)=
\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma}
\left(1+\frac{(r-m)^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}.
```

The scale parameter determines the standard deviation. For $\nu>2$,
$\operatorname{sd}(r)=\sigma\sqrt{\nu/(\nu-2)}$.
The heavy tails allow occasional extreme returns without treating each one as
conclusive evidence of a mean change.

`core/model.py` estimates $(\mu,\sigma,\nu)$ from the warm-up. It searches 25
geometrically spaced $\nu$ values from 2.05 to 50 and performs 50 EM updates of
location and scale for each value, selecting the highest likelihood. This is a
finite-grid approximation to a continuous maximum-likelihood fit.

## 2. Daily evidence: the log-likelihood ratio

Define evidence in favor of the broken model by

```math
\ell_t=\log\frac{f(r_t;\mu_b,\sigma,\nu)}{f(r_t;\mu,\sigma,\nu)}.
```

The common normalization cancels, giving

```math
\ell_t=\frac{\nu+1}{2}\log
\frac{1+(r_t-\mu)^2/(\nu\sigma^2)}
{1+(r_t-\mu_b)^2/(\nu\sigma^2)}.
```

Positive evidence favors deterioration; negative evidence favors the healthy
model. `log_likelihood_ratio()` subtracts Student-t log densities to avoid
forming a potentially unstable ratio of small probabilities.

For fixed correct models with independent observations,

```math
\mathbb E_0[\ell_t]=-D_{\mathrm{KL}}(f_0\Vert f_1)\leq0.
```

```math
\mathbb E_1[\ell_t]=D_{\mathrm{KL}}(f_1\Vert f_0)\geq0.
```

Thus the statistic tends downward under the healthy model and upward under
the broken model.

## 3. CUSUM recurrence and alarm

Start with $S_0=0$ and update

```math
S_t=\max(0,S_{t-1}+\ell_t).
```

Equivalently,

```math
S_t=\max\left(0,\max_{1\leq j\leq t}\sum_{i=j}^{t}\ell_i\right).
```

For conditionally independent observations, each suffix sum is the log
likelihood ratio for a change beginning at $j$. CUSUM retains the strongest
nonnegative evidence over candidate change times. The floor at zero prevents a
long healthy history from creating a large negative balance that must be repaid.

An alarm occurs at

```math
\tau_c=\inf\{t\geq1:S_t>c\}.
```

The **strict inequality** matches Killswitch. With $c=2$, a statistic of exactly
2 does not fire. Both the statistic $S_t$ and its threshold are measured in
natural-log evidence units.

For daily evidence $[1,-0.5,-1,2]$:

| Day | Evidence $\ell_t$ | Statistic $S_t$ |
|---|---:|---:|
| 1 | 1.0 | 1.0 |
| 2 | -0.5 | 0.5 |
| 3 | -1.0 | 0.0 |
| 4 | 2.0 | 2.0 |

A larger threshold generally trades fewer false alarms for slower detection.
The specified healthy and broken models determine the log-likelihood increments.

## 4. Threshold selection and operational policy

The original project chooses $c$ by simulated regret, rather than imposing a
particular false-alarm probability. This extraction retains that procedure:

1. Generate 300 training strategies with seed 1: 100 healthy, 100 sudden mean
   flips, and 100 gradual decays. Each has 750 days and a 250-day warm-up.
2. Fit each strategy's baseline on its own warm-up and calculate its evidence.
3. Evaluate 59 thresholds from 0.5 to 15 in increments of 0.25.
4. Choose the threshold with minimum average training regret.
5. Evaluate that fixed threshold on 300 fresh strategies with seed 2.

For the demo, simulation parameters are $\mu=0.0015$, $\sigma=0.008$, and $\nu=4$.
The original real-data pipeline instead simulates from the observed warm-up
fit and matches the monitoring horizon. The standalone demo reproduces the
synthetic case. The [original Killswitch project](https://github.com/leilei67/Killswitch)
contains the web interface and real-data pipeline.

Let $a_t^{(c)}$ indicate whether the rule trades on day $t$, and let $b$ be the
true simulated break day. Using **additive** return accounting from day 250,

```math
R(c)=\sum_{t=250}^{T-1}r_t\,\mathbf 1\{t<b\}
-\sum_{t=250}^{T-1}r_t\,a_t^{(c)}.
```

```math
\widehat c=\operatorname*{arg\,min}_{c\in\mathcal C}
\frac1N\sum_{n=1}^N R_n(c).
```

The oracle stops at the true break. The rule observes the return at the close,
so it **trades on the alarm day**, sits out the following 60 days, and restarts
with $S=0$. An alarm on zero-based day $t$ means trading resumes on $t+61$.
The baseline is kept fixed; restarting does not refit it. Gradual decay crosses
zero halfway through its 120-day transition, which defines its true break.

`cusum()` gives a continuous statistic with no resets at alarms. The reset and
sit-out behavior belongs to `run_kill_rule()`. These serve different purposes.
In the original application, CUSUM crossing its threshold recommends **pause**
when the Bayesian kill condition has not already taken precedence; the generic
simulator evaluates CUSUM as its own stop-and-restart rule.

False kills count strategies with at least one alarm before their true break
(including healthy strategies). Missed breaks count broken strategies with no
alarm on or after the break. Delay averages the first post-break alarm's delay
across detected strategies only. Together these metrics describe the trade-off
between false alarms, missed breaks, detection speed, and trading performance.

## Files and verification

| File | Purpose |
|---|---|
| [core/detectors.py](core/detectors.py) | CUSUM recurrence, strict alarm comparison, batched series |
| [core/model.py](core/model.py) | Student-t fitting and daily log-likelihood ratio |
| [core/sim.py](core/sim.py) | Healthy, sudden-break, and decay simulation |
| [core/evaluate.py](core/evaluate.py) | Stop/restart execution, regret metrics, grid tuning |
| [core/demo.py](core/demo.py) | Reproducible CUSUM-only training/test demonstration |
| [tests/](tests/) | Mathematical identities, model fitting, simulations, restart timing |
| [ATTRIBUTION.md](ATTRIBUTION.md) | Team credit and exact upstream provenance |

The tests compare the recurrence with the maximum-suffix formula, compare daily
evidence with independent SciPy densities and the closed form above, check
strict threshold behavior and 60-day restarts, and retain upstream model and
simulation tests. The lockfile records the dependency versions used here.

## Model assumptions

The baseline uses the first 250 days as a representative healthy period with a
positive fitted mean. The alternative models deterioration as a change to a
negative mean, keeping the Student-t scale and tail shape fixed. Likelihoods
use independent daily observations with fixed fitted parameters. The threshold
is selected for the simulation horizon and regret objective described above.

## Team and source

This detector comes from our collaborative **Killswitch** project. Other team
members are **Leiyi Sun** and **Zijian Wang**. See [ATTRIBUTION.md](ATTRIBUTION.md)
for source links and extraction details.
