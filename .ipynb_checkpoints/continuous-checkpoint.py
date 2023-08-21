import itertools
from qutip import *
import numpy as np
from measurements import *
import abc
import scipy.sparse as sp
from math import factorial
from qudit import *

class ContinuousProtocol:
    def __init__(self, base_state: Qobj, res, m_i, m_c):
        self.base_state = base_state
        self.N = base_state.dims[0][0]
        self.res = res
        self.m_i = m_i
        self.m_c = m_c
        self.decode_base_list = self._create_base_list()

    def simulate_fidelity_specific(self, A1, B1, A2, B2, gamma_loss, gamma_dephasing, decode_res=10):
        initial_state = self._create_initial_state()
        noisy_state = self._add_noise(initial_state, gamma_loss, gamma_dephasing)
        M1 = tensor(self._create_phase_parity_measurement(A1), self._create_phase_parity_measurement(B1))
        M2 = tensor(self._create_photon_parity_measurement(A2), self._create_photon_parity_measurement(B2))
        U1 = self._create_phase_rotation(A1, B1)
        # plot_wigner((M1 * noisy_state * M1.dag()).ptrace(0))
        state1 = M2 * M1 * noisy_state * M1.dag() * M2.dag()
        print("before rotation")
        # plot_wigner(state1.ptrace(0))
        # plot_wigner(state1.ptrace(1))
        self.final = U1 * state1 * U1.dag()
        # print("after rotation")
        # plot_wigner(state2.ptrace(0))
        # plot_wigner(state2.ptrace(1))


        # this is the decoding process. not the same resolution as the self.res!
        self.BosonicObject = EntangledBosonicQudit(self.N, self.m_c, res=decode_res, d2=None,
                                                   base_state_list=self.decode_base_list)
        self.decoded_qudit = self.BosonicObject.cavity_to_entangled_qudits(self.final)

        # psi_p = (tensor(basis(2,1),basis(2,0)) + tensor(basis(2,0),basis(2,1))).unit()
        # self.decoded_qudit.dims = [[2]*self.m_c]*2
        # print(fidelity(tt.ptrace([0,2]),psi_p)**2)
        
        # U2 = self._create_photon_rotation(A2, B2)
        ## now should do rotation and fidelity measurement

    def _create_initial_state(self):
        rotation = lambda theta: (theta * 1.0j * num(self.N)).expm()
        return sum([tensor(rotation(i * 2 * np.pi / self.m_i) * self.base_state,
                           rotation(i * 2 * np.pi / self.m_i) * self.base_state)
                    for i in range(self.m_i)]).unit()

    def _add_noise(self, state: Qobj, gamma_loss, gamma_dephasing, max_loss_level = 10):
        """
        adding noise to the state. first to cavity A, cavity B, then dephasing.
        """
        if state.isket:
            state = ket2dm(state)
        kraus_list = NoiseChannels.get_loss_kraus_list(self.N, gamma_loss, max_level=max_loss_level)
        dephasing_super = NoiseChannels.get_dephasing_channel(self.N, gamma_dephasing)
        dephasing_tensor = super_tensor(dephasing_super, dephasing_super)
        I = qeye(self.N)
        A_loss = sum([tensor(kraus_list[i], I) * state * tensor(kraus_list[i], I).dag()
                      for i in range(max_loss_level)])
        AB_loss = sum([tensor(I, kraus_list[i]) * A_loss * tensor(I, kraus_list[i]).dag()
                       for i in range(max_loss_level)])
        return dephasing_tensor(AB_loss)

    def _create_phase_parity_measurement(self, result: int):
        """
        :param result: an int from 0 to m_c-1, which stands for A1 or B1 result.
        """
        assert(0 <= result < self.m_c)
        delta_c = self.res//self.m_c
        N = self.N  # just for readability
        M_class = CanonicalPhaseMeasurement(1 / np.sqrt(N) * sum([basis(N, i) for i in range(N)]), self.res, num(N))
        return sum(M_class.POVM_elements[M_class.get_possible_results()[result + i * delta_c]] for i in range(self.m_c))

    def _create_photon_parity_measurement(self, result):
        # delta_c = self.res//self.m_c
        return sum([ket2dm(basis(self.N, i)) for i in range(result, self.N, self.m_c//2)])

    def _create_phase_rotation(self, A1, B1):
        delta_c = self.res // self.m_c
        rotation = lambda theta: (theta * 1.0j * num(self.N)).expm()
        U_A = rotation(-A1 / self.res * 2 * np.pi)
        s_B = min([B1 - delta_c, B1, B1 + delta_c], key=lambda x: abs(A1-x))
        print(f"{s_B=}")
        U_B = rotation(-s_B / self.res * 2 * np.pi)
        return tensor(U_A, U_B)

    def _create_photon_rotation(self, A1, B1):
        raise NotImplementedError

    def _create_base_list(self):
        return [sum([basis(self.N, i) for i in range(k, self.N, self.m_c)]).unit() for k in range(self.m_c)]


class NoiseChannels:
    @staticmethod
    def get_dephasing_channel(N, gamma_dephasing):
        # Create a sparse diagonal matrix
        sparse_matrix = sp.diags(np.array([np.exp(-gamma_dephasing / 2 * (m - n) ** 2)
                                           for m in range(N) for n in range(N)]), format="csr")
        return qt.Qobj(sparse_matrix, dims=[[[N], [N]], [[N], [N]]])

    @staticmethod
    def get_loss_kraus(N, gamma_loss, level):
        # N = rho.dims[0][0]
        one_minus_gamma_to_n = qt.Qobj(sp.diags((1-gamma_loss) ** (num(N).diag()/2)))
        if level == 0:
            return one_minus_gamma_to_n
        return np.sqrt(gamma_loss ** level / factorial(level)) * one_minus_gamma_to_n * destroy(N)**level

    @staticmethod
    def get_loss_kraus_list(N, gamma_loss, max_level=10):
        return [NoiseChannels.get_loss_kraus(N, gamma_loss, level) for level in range(max_level)]
