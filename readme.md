The plots for figure 4 and 5 is from `discrete_simulations.ipynb`
The plot for figure 6 and 16 is from `abort.ipynb`
The plot for figure 8 is from `break_even_gamma_ratio_alpha7_photon_improved.ipynb`
The plot for figure 9, 11 and 12 is from `delta_alpha_squeezing_grid.ipynb`

## Installation

Python 3.9, 3.10, or 3.11 is required. From the repository root, create a
virtual environment and install the project with `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To run the notebooks, install the optional notebook dependencies and a Jupyter
frontend instead:

```bash
python -m pip install -e ".[notebooks]" jupyterlab
jupyter lab
```

## Photon-distribution-aware discrete loss model

`PhotonDistributionQudit` is an opt-in replacement for the loss probabilities
used by `Qudit`. It accepts a single-mode Fock probability distribution and
uses the exact binomial average for each photon-loss count:

```python
from discrete_simulation import DiscreteSimulation
from qudit import PhotonDistributionQudit, photon_number_distribution_from_state

photon_distribution = photon_number_distribution_from_state(state_before_loss)
simulation = DiscreteSimulation(
    d,
    m_i,
    m_c,
    qudit_class=PhotonDistributionQudit,
    qudit_kwargs={"photon_number_distribution": photon_distribution},
)
```

Physical loss counts are aggregated modulo `d`. The two cavities use the same
marginal distribution independently, matching the existing discrete model's
assumption. This improves loss-count probabilities, but it does not model
conditional-state deformation from the full loss Kraus action.
