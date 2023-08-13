import itertools
from qutip import *
import numpy as np
from measurements import *

class ContinuousProtocol:
    def __init__(self, base_state: Qobj, res, m_i, m_c):
        self.base_state = base_state
        self.N = base_state.dims[0][0]
        self.res = res
        self.m_i = m_i
        self.m_c = m_c

    def simulate(self, A_1, B_1, A_2, B_2, gamma_loss, gamma_dephasing):
        initial_state = self._create_initial_state()
        noisy_state = self._add_noise(initial_state)
        M1 = tensor(self._create_phase_parity_measurement(A_1),self._create_phase_parity_measurement(B_1))
        M2 = tensor(self._create_photon_parity_measurement(A_2), self._create_phase_parity_measurement(B_2))
        state1 = M2 * M1 * noisy_state * M1.dagger() * M2.dagger()
        ## now should do rotation and fidelity measurement

    def _create_initial_state(self):
        rotation = lambda theta: (theta * 1.0j * num(N)).expm()
        return sum([tensor(rotation(i * 2 * np.pi / self.m_i) * self.base_state,
                           rotation(i * 2 * np.pi / self.m_i) * self.base_state)
                    for i in range(self.m_i)])


    def _add_noise(self, state: Qobj, tk, tk_phi):
        """
        This is not true!!!!!!!!
        """
        if state.isket():
            state = ket2dm(state)
        E_tk = np.sqrt(tk) * destroy(N)
        E_tk_phi = np.sqrt(tk_phi) * num(N)
        E_0 = qeye(N) - 1 / 2 * E_tk.dag() * E_tk - 1 / 2 * E_tk_phi.dag() * E_tk_phi
        new_state = E_0.dag() * dm_state * E_0 + E_tk.dag() * dm_state * E_tk + E_tk_phi.dag() * dm_state * E_tk_phi
        return new_state

    def _create_phase_parity_measurement(self, result: int):
        """
        :param result: an int from 0 to m_c-1, which stands for A1 or B1 result.
        """
        assert(0 <= result < self.m_c)
        delta_c = self.res/self.m_c
        N = self.N  # just for readability
        M_class = CanonicalPhaseMeasurement(1 / np.sqrt(N) * sum([basis(N, i) for i in range(N)]), self.res, num(N))
        return sum(M_class.POVM_elements[M_class.get_possible_results()[result + i * delta_c]] for i in range(self.m_c))

    def _create_photon_parity_measurement(self, result):
        raise NotImplementedError