import qutip
import numpy as np

from State import State
from Operator import Operator
from Simulation import Simulation
from QutipWrapper import QutipWrapper


class SimulationDiscreteErrorModel(Simulation):
    def __init__(
            self,
            number_of_fock_states: int = 80,
            number_of_rotations: int = 4,
            initial_state_name: str = 'pegg-barnett',
            rotation_probability: float = 0.05,
            *args,
            **kwargs):
        self.rotation_probability = rotation_probability

        super().__init__(
            number_of_fock_states=number_of_fock_states,
            number_of_rotations=number_of_rotations,
            number_of_parties=2,
            initial_state_name=initial_state_name,
            *args,
            **kwargs)

    def _add_noise(self) -> State:
        rotation_operator_base = Operator.create('rotation',
                                                 self.number_of_fock_states,
                                                 angle_in_radians=2 * np.pi / self.number_of_rotations)

        phi_minus = self.initial_state.apply_operator(
            QutipWrapper.tensor(rotation_operator_base, qutip.qeye(self.number_of_fock_states)))
        phi_plus = self.initial_state.apply_operator(
            QutipWrapper.tensor(qutip.qeye(self.number_of_fock_states), rotation_operator_base))
        phi_2 = self.initial_state.apply_operator(
            QutipWrapper.tensor(rotation_operator_base ** 2, qutip.qeye(self.number_of_fock_states)))

        self.noisy_state = State((1 - 2 * self.rotation_probability - self.rotation_probability ** 2)
                                 * qutip.ket2dm(self.initial_state)
                                 + self.rotation_probability * qutip.ket2dm(phi_minus)
                                 + self.rotation_probability * qutip.ket2dm(phi_plus)
                                 + self.rotation_probability ** 2 * qutip.ket2dm(phi_2))