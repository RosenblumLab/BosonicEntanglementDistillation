import itertools
from qutip import *
import numpy as np

class ContinuousProtocol:
    def __init__(self, base_state: Qobj, res, m_i, m_c):
        self.base_state = base_state
        self.N = base_state.dims[0][0]
        self.res = res
        self.m_i = m_i
        self.m_c = m_c

    def simulate(self, A_1, B_1, A_2, B_2):
        initial_state = self._create_initial_state()
        noisy_state = self._add_noise(initial_state)
        M1 = tensor(self._create_phase_parity_measurement(A_1),self._create_phase_parity_measurement(B_1))
        M2 = tensor(self._create_photon_parity_measurement(A_2), self._create_phase_parity_measurement(B_2))
        state1 = M2 * M1 * noisy_state * M1.dagger() * M2.dagger()
        ## now should do rotation and fidelity measurement


    def _create_initial_state(self):
        raise NotImplementedError

    def _add_noise(self, state: Qobj):
        raise NotImplementedError

    def _create_phase_parity_measurement(self,result):
        raise NotImplementedError

    def _create_photon_parity_measurement(self, result):
        raise NotImplementedError