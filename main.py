import qutip
import numpy as np
import matplotlib.pyplot as plt

from SimulationContinuousErrorModel import SimulationContinuousErrorModel

if __name__ == '__main__':
    number_of_fock_states = 30
    number_of_parties = 2
    number_of_rotations = 4
    number_of_parts_to_decode = 2
    initial_state_name = 'pegg-barnett'

    reference_state = np.zeros(
       [number_of_parts_to_decode ** number_of_parties, number_of_parts_to_decode ** number_of_parties])
    reference_state[0, 0] = 0.5
    reference_state[-1, -1] = 0.5
    reference_state[0, -1] = 0.5
    reference_state[-1, 0] = 0.5
    reference_state = qutip.Qobj(reference_state)

    simulation = SimulationContinuousErrorModel(
       number_of_fock_states=number_of_fock_states,
       number_of_rotations=number_of_rotations,
       number_of_parties=number_of_parties,
       initial_state_name=initial_state_name,
       kappa_dephase=0.1,
       kappa_decay=0.1)

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(10)
    fig.set_figheight(4)

    qutip.plot_wigner(simulation.initial_state.ptrace(0), fig, axes[0], colorbar=True)

    qutip.plot_wigner(simulation.noisy_state.ptrace(0), fig, axes[1], colorbar=True)

    plt.show()