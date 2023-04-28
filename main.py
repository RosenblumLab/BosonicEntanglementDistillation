import qutip
import numpy as np
import matplotlib.pyplot as plt

from State import State
from Operator import Operator
from QutipWrapper import QutipWrapper

from SimulationContinuousErrorModel import SimulationContinuousErrorModel

if __name__ == '__main__':
    number_of_fock_states = 30
    number_of_parties = 2
    number_of_rotations = 4
    initial_state_name = 'pegg-barnett'

    # simulation = SimulationContinuousErrorModel(
    #     number_of_fock_states=number_of_fock_states,
    #     number_of_rotations=number_of_rotations,
    #     initial_state_name=initial_state_name,
    #     number_of_parity_sectors=4,
    #     kappa_decay=0,
    #     kappa_dephase=0)
    state = State.create(
        'pegg-barnett',
        number_of_fock_states=30,
        number_of_parties=2,
        number_of_rotations=4,
        rotate_before_sum=False)

    decay_base = np.sqrt(0.1) * Operator.create('destroy', 30)
    dephase_base = np.sqrt(0.1) * Operator.create('number', 30)

    noisy_state = State.mesolve(
        QutipWrapper.repeat(qutip.qzero(30), 2),
        state,
        [0, 1],
        [qutip.tensor(decay_base, qutip.qeye(30)),
         qutip.tensor(dephase_base, qutip.qeye(30)),
         qutip.tensor(qutip.qeye(30), decay_base),
         qutip.tensor(qutip.qeye(30), dephase_base)]
    )[-1].unit()

    wigner = Operator.create(
        name='wigner-parity',
        number_of_fock_states=30,
        number_of_parties=2,
        number_of_parity_sectors=4)

    qutip.plot_wigner(noisy_state.evaluate_operator(wigner).ptrace(0))

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)

    # qutip.plot_wigner(simulation.initial_state.ptrace(0), fig, axes[0], colorbar=True)
    #
    # qutip.plot_wigner(simulation.state_after_protocol.ptrace(0), fig, axes[1], colorbar=True)

    plt.show()