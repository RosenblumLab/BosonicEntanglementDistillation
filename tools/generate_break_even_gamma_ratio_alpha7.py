#!/usr/bin/env python
"""Generate cached data for alpha=7 break-even plots.

Run from the repository root:

    python tools/generate_break_even_gamma_ratio_alpha7.py

The script writes one pickle per gamma_l/gamma_phi ratio under results/.
It checkpoints after each expensive component, so rerunning resumes missing
work unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import multiprocess as mp
import numpy as np
from qutip import coherent, displace, squeeze
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continuous import ContinuousProtocol
from discrete_simulation import ContinuousSimulation, DiscreteSimulation

RESULTS_DIR = ROOT / "results"

D = 16
M_I = 16
M_C = 8
M_F = 2
ALPHA = 7
N = 200
DECODE_RES = 8
SQUEEZED = -0.8
FIDELITY_CUT = 0
DEFAULT_GAMMA_LOSS_MIN = 1e-3
DEFAULT_GAMMA_LOSS_MAX = 1e-1
DEFAULT_GAMMA_LOSS_POINTS = 15


def ratio_label(ratio: float) -> str:
    return f"{ratio:g}".replace(".", "p")


def gamma_range_label(gamma_loss_list: np.ndarray) -> str:
    default_gamma_loss_list = np.logspace(
        np.log10(DEFAULT_GAMMA_LOSS_MIN),
        np.log10(DEFAULT_GAMMA_LOSS_MAX),
        DEFAULT_GAMMA_LOSS_POINTS,
    )
    if np.array_equal(gamma_loss_list, default_gamma_loss_list):
        return ""

    gamma_min = f"{gamma_loss_list[0]:.0e}".replace("-", "m")
    gamma_max = f"{gamma_loss_list[-1]:.0e}".replace("-", "m")
    return f"_gamma_l={gamma_min}_{gamma_max}_points={len(gamma_loss_list)}"


def cache_path(ratio: float, gamma_loss_list: np.ndarray) -> Path:
    return (
        RESULTS_DIR
        / f"break_even_alpha=7_gamma_l_over_gamma_phi={ratio_label(ratio)}{gamma_range_label(gamma_loss_list)}.pkl"
    )


def gamma_pairs_for_ratio(ratio: float, gamma_loss_list: np.ndarray) -> list[tuple[float, float]]:
    gamma_dephasing_list = gamma_loss_list / ratio
    return list(zip(gamma_loss_list, gamma_dephasing_list))


def poisson_std_for_state(state) -> float:
    fock_dist = np.abs(np.array(state)) ** 2
    photon_numbers = np.arange(len(fock_dist))
    mean = np.sum(photon_numbers * fock_dist.transpose())
    return np.sqrt(np.sum(fock_dist.transpose() * (photon_numbers - mean) ** 2)).item()


def compute_state_metadata() -> dict[str, float]:
    coherent_state = coherent(N, ALPHA)
    pro_obj = ContinuousProtocol(coherent_state, res=D, m_i=M_I, m_c=M_C)
    coherent_limit = (
        coherent_state.dag()
        * pro_obj._create_phase_parity_measurement(1)
        * coherent_state
    ).full()[0][0].real

    squeezed_state = displace(N, ALPHA) * squeeze(N, SQUEEZED) * coherent(N, 0)
    squeezed_limit = (
        squeezed_state.dag()
        * pro_obj._create_phase_parity_measurement(1)
        * squeezed_state
    ).full()[0][0].real

    return {
        "limit": coherent_limit,
        "limit_sq": squeezed_limit,
        "std_coherent": poisson_std_for_state(coherent_state),
        "std_squeezed": poisson_std_for_state(squeezed_state),
    }


def load_cached(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as file:
        return dill.load(file)


def save_cached(path: Path, data: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with path.open("wb") as file:
        dill.dump(data, file)


def run_discrete(
    alpha_for_discrete: float,
    ratio: float,
    processes: int,
    gamma_loss_list: np.ndarray,
):
    sim = DiscreteSimulation(D, M_I, M_C, m_f=M_F, alpha=alpha_for_discrete)
    with mp.Pool(processes) as pool:
        fid_prob = pool.starmap(
            sim.average_fidelity,
            [(gl, gp, FIDELITY_CUT) for gl, gp in gamma_pairs_for_ratio(ratio, gamma_loss_list)],
        )
    return [tup[0] for tup in fid_prob], [tup[1] for tup in fid_prob]


def run_continuous(
    ratio: float,
    gamma_loss_list: np.ndarray,
    processes: int,
    *,
    use_squeezed: bool,
):
    sim = ContinuousSimulation(
        N,
        D,
        M_I,
        M_C,
        m_f=M_F,
        alpha=ALPHA,
        decode_res=DECODE_RES,
        squeezed=SQUEEZED if use_squeezed else 0,
    )
    jobs = [
        (gamma_loss, gamma_dephasing, FIDELITY_CUT)
        for gamma_loss, gamma_dephasing in gamma_pairs_for_ratio(ratio, gamma_loss_list)
    ]
    description = f"continuous squeezed={use_squeezed}, ratio={ratio:g}"
    with mp.Pool(processes) as pool:
        fid_prob = list(
            tqdm(
                pool.starmap(sim.average_fidelity, jobs),
                total=len(jobs),
                desc=description,
            )
        )
    return [tup[0] for tup in fid_prob], [tup[1] for tup in fid_prob]


def compute_or_resume_ratio(
    ratio: float,
    *,
    force: bool,
    discrete_processes: int,
    continuous_processes: int,
    metadata: dict[str, float],
    gamma_loss_list: np.ndarray,
) -> dict:
    path = cache_path(ratio, gamma_loss_list)
    data = {} if force else load_cached(path)
    data.update(
        {
            "ratio": ratio,
            "gamma_loss_list": gamma_loss_list,
            "gamma_dephasing_list": gamma_loss_list / ratio,
            **metadata,
        }
    )
    save_cached(path, data)

    steps = [
        (
            ("fid_discrete", "prob_discrete"),
            lambda: run_discrete(ALPHA, ratio, discrete_processes, gamma_loss_list),
        ),
        (
            ("fid_discrete_minus", "_prob_discrete_minus"),
            lambda: run_discrete(
                np.sqrt(ALPHA**2 - metadata["std_coherent"]),
                ratio,
                discrete_processes,
                gamma_loss_list,
            ),
        ),
        (
            ("fid_discrete_plus", "_prob_discrete_plus"),
            lambda: run_discrete(
                np.sqrt(ALPHA**2 + metadata["std_coherent"]),
                ratio,
                discrete_processes,
                gamma_loss_list,
            ),
        ),
        (
            ("fid_discrete_minus_sq", "_prob_discrete_minus_sq"),
            lambda: run_discrete(
                np.sqrt(ALPHA**2 - metadata["std_squeezed"]),
                ratio,
                discrete_processes,
                gamma_loss_list,
            ),
        ),
        (
            ("fid_discrete_plus_sq", "_prob_discrete_plus_sq"),
            lambda: run_discrete(
                np.sqrt(ALPHA**2 + metadata["std_squeezed"]),
                ratio,
                discrete_processes,
                gamma_loss_list,
            ),
        ),
        (
            ("fid_continuous", "prob_continuous"),
            lambda: run_continuous(
                ratio,
                gamma_loss_list,
                continuous_processes,
                use_squeezed=False,
            ),
        ),
        (
            ("fid_continuous_sq", "prob_continuous_sq"),
            lambda: run_continuous(
                ratio,
                gamma_loss_list,
                continuous_processes,
                use_squeezed=True,
            ),
        ),
    ]

    for keys, runner in steps:
        if all(key in data for key in keys):
            print(f"Skipping {keys[0]} for gamma_l/gamma_phi={ratio:g}; already cached.")
            continue

        print(f"Computing {keys[0]} for gamma_l/gamma_phi={ratio:g}")
        first, second = runner()
        data[keys[0]] = first
        data[keys[1]] = second
        save_cached(path, data)

    print(f"Saved {path.relative_to(ROOT)}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cached alpha=7 break-even data for several gamma_l/gamma_phi ratios."
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
        help="gamma_l/gamma_phi ratios to generate.",
    )
    parser.add_argument(
        "--discrete-processes",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes for discrete simulations.",
    )
    parser.add_argument(
        "--continuous-processes",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes for continuous simulations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all components even when cached data exists.",
    )
    parser.add_argument(
        "--gamma-loss-min",
        type=float,
        default=DEFAULT_GAMMA_LOSS_MIN,
        help="Minimum gamma_l value.",
    )
    parser.add_argument(
        "--gamma-loss-max",
        type=float,
        default=DEFAULT_GAMMA_LOSS_MAX,
        help="Maximum gamma_l value.",
    )
    parser.add_argument(
        "--gamma-loss-points",
        type=int,
        default=DEFAULT_GAMMA_LOSS_POINTS,
        help="Number of log-spaced gamma_l values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gamma_loss_list = np.logspace(
        np.log10(args.gamma_loss_min),
        np.log10(args.gamma_loss_max),
        args.gamma_loss_points,
    )
    metadata = compute_state_metadata()
    for ratio in args.ratios:
        compute_or_resume_ratio(
            ratio,
            force=args.force,
            discrete_processes=args.discrete_processes,
            continuous_processes=args.continuous_processes,
            metadata=metadata,
            gamma_loss_list=gamma_loss_list,
        )


if __name__ == "__main__":
    main()
