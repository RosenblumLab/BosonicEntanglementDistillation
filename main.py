import qutip

import numpy as np

from DecodeUtils import DecodeUtils
from SimulationContinuousErrorModel import SimulationContinuousErrorModel

if __name__ == '__main__':
    number_of_fock_states = 30
    number_of_parties = 2
    number_of_rotations = 4
    initial_state_name = 'pegg-barnett'
    number_of_parts_to_decode = 2

    simulation = SimulationContinuousErrorModel(
        number_of_fock_states=number_of_fock_states,
        number_of_rotations=number_of_rotations,
        initial_state_name=initial_state_name,
        number_of_parties=number_of_parties,
        kappa_dephase=0,
        kappa_decay=0.5)

    reference_state = np.zeros(
        [number_of_parts_to_decode ** number_of_parties, number_of_parts_to_decode ** number_of_parties])
    reference_state[0, 0] = 0.5
    reference_state[-1, -1] = 0.5
    reference_state[0, -1] = 0.5
    reference_state[-1, 0] = 0.5
    reference_state = qutip.Qobj(reference_state)

    decoded_state_after_protocol = DecodeUtils.decode(
        simulation.state_after_protocol,
        number_of_fock_states=number_of_fock_states,
        number_of_parties=number_of_parties,
        number_of_parts=number_of_parts_to_decode)
