import qutip
import numpy as np

from State import State
from Operator import Operator
from Simulation import Simulation


class SimulationContinuousErrorModel(Simulation):
    def __init__(
            self,
            number_of_fock_states: int = 80,
            number_of_rotations: int = 4,
            number_of_parties: int = 1,
            initial_state_name: str = 'pegg-barnett',
            number_of_parity_sectors: int = 4,
            kappa_decay: float = 0.01,
            kappa_dephase: float = 0.01,
            *args,
            **kwargs):

        self.kappa_decay = kappa_decay
        self.kappa_dephase = kappa_dephase

        super().__init__(
            number_of_fock_states=number_of_fock_states,
            number_of_rotations=number_of_rotations,
            number_of_parties=number_of_parties,
            initial_state_name=initial_state_name,
            number_of_parity_sectors=number_of_parity_sectors,
            *args,
            **kwargs)

    def _get_indexed_collapse_operator(self, noise_operator: qutip.Qobj, index: int) -> qutip.Qobj:
        operator_list = index \
                        * [qutip.qeye(self.number_of_fock_states)] \
                        + [noise_operator] \
                        + (self.number_of_parties - index - 1) * [qutip.qeye(self.number_of_fock_states)]
        return qutip.tensor(operator_list)

    def _get_collapse_operators(self) -> list[qutip.Qobj]:
        decay_base = np.sqrt(self.kappa_decay) * Operator.create('destroy', self.number_of_fock_states)
        dephase_base = np.sqrt(self.kappa_dephase) * Operator.create('destroy', self.number_of_fock_states)

        collapse_operators = []

        for i in range(self.number_of_parties):
            decay_collapse_operator = self._get_indexed_collapse_operator(decay_base, i)
            dephase_collapse_operator = self._get_indexed_collapse_operator(dephase_base, i)
            collapse_operators.extend([decay_collapse_operator, dephase_collapse_operator])

        return collapse_operators

    def _solve_master_equation(self, t_list: list[float] = [0, 1]) -> State:
        collapse_operators = self._get_collapse_operators()

        hamiltonian = qutip.tensor(
            [qutip.qzero(self.number_of_fock_states)] * self.number_of_parties)

        return State.mesolve(hamiltonian, self.initial_state, t_list, collapse_operators)

    def _add_noise(self):
        self.noisy_state = self._solve_master_equation()[-1].unit()
