import qutip
import numpy as np
import matplotlib.pyplot as plt

from State import State
from Operator import Operator
from QutipWrapper import QutipWrapper

from SimulationDiscreteErrorModel import SimulationDiscreteErrorModel

from SimulationContinuousErrorModel import SimulationContinuousErrorModel
from Simulation import Simulation

if __name__ == '__main__':
    number_of_fock_states = 30
    number_of_parties = 2
    number_of_rotations = 4
    initial_state_name = 'pegg-barnett'

    simulation = SimulationDiscreteErrorModel(
        number_of_fock_states=number_of_fock_states,
        number_of_rotations=number_of_rotations,
        initial_state_name=initial_state_name,
        rotation_probability=0.05)

    fig, axes = plt.subplots(1, 3)
    fig.set_figwidth(14)
    fig.set_figheight(4)

    qutip.plot_wigner(simulation.initial_state.ptrace(0), fig, axes[0], colorbar=True)

    qutip.plot_wigner(simulation.noisy_state.ptrace(0), fig, axes[1], colorbar=True)

    qutip.plot_wigner(simulation.state_after_protocol.ptrace(0), fig, axes[2], colorbar=True)


    plt.show()