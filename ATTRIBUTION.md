# Team credit and provenance

This repository presents the CUSUM component of the collaborative Killswitch
project. Other members of the team are **Leiyi Sun** and **Zijian Wang**.
The original implementation is credited to the Killswitch team.

Original repository: [Killswitch](https://github.com/leilei67/Killswitch)

Source commit: [`83e98fbbef3ce299ee087eb7bd3f80c43ecd7300`](https://github.com/leilei67/Killswitch/tree/83e98fbbef3ce299ee087eb7bd3f80c43ecd7300)

- `core/model.py`, `core/sim.py`, `core/evaluate.py`, `core/__init__.py`,
  `tests/test_model.py`, and `tests/test_sim.py` are copied from that commit.
- `core/detectors.py` keeps the upstream `Cusum` class behavior and adapts the
  upstream `cusum`/`run_series` wrapper into a CUSUM-only function. Other detectors
  and the SciPy `expit` dependency are omitted.
- `core/demo.py` adapts the upstream pipeline's CUSUM training and evaluation.
- The mathematical walkthrough, extraction tests, packaging, and standalone
  presentation were prepared for this repository.
