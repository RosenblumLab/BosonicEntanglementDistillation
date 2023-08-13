import itertools
from qutip import *
import numpy as np
from scipy.special import erf
from functools import lru_cache
# from memory_profiler import profile

from scipy import stats

class BosonicQudit:
    def __init__(self, N, d, res=1000, base_state_list=None):
        self.N = N
        self.d = d
        self.res = res
        self.proj_op_list = self.create_proj_list()
        if base_state_list is None:
            pegg_bernet_0 = 1 / np.sqrt(N) * sum([basis(N, i) for i in range(N)])
            rotation = lambda x: (x * 1.0j * num(N)).expm()
            base_state_list = [rotation(2 * i * np.pi / d) * pegg_bernet_0 for i in range(d)]
        self.basis_dict, self.phi_list = self.create_basis_dictionary(self.res, base_state_list)

    def cavity_to_qudit(self, rho):
        """
        Integrating over <phi|rho|phi> in order to partial trace the lo part
        :param rho: the density matrix of the cavity
        :return: sigma, density matrix of the qudit
        """
        d = self.d
        sigma = np.zeros([d, d])
        dphi = 2 * np.pi / d / self.res
        for i, j in itertools.product(range(d), range(d)):
            sigma[i, j] = dphi * sum([self.basis_dict[(i, phi)].dag() * rho * self.basis_dict[(j, phi)]
                                      for phi in self.phi_list])[0][0][0]
        return Qobj(sigma).unit()

    def create_proj_list(self):
        op_list = []
        d = self.d
        N = self.N
        for i in range(d):
            phi_plus = 2 * i * np.pi/d + np.pi / d
            phi_minus = 2 * i * np.pi/d - np.pi / d
            proj = Qobj(np.array([[(np.exp(1j * (m - n) * phi_plus) - np.exp(1j * (m - n) * phi_minus)) / (
                        1j * (m - n)) if m != n else (phi_plus - phi_minus) for n in range(N)] for m in range(N)]))
            op_list.append(proj)
        return op_list

    def create_basis_dictionary(self, res, base_state_list):
        """
        creates a dictionary that gets a tuple (i,phi) and returns the relevant vector
        :res: the resolution in which we work at
        :return: Dictionary of kets.
        """
        N = self.N
        d = self.d
        assert(len(base_state_list) == d)

        rotation = lambda x: (x * 1.0j * num(N)).expm()
        phi_minus = -np.pi/d
        phi_plus = np.pi/d
        phi_list = np.linspace(phi_minus, phi_plus, res)
        basis_dict = {}
        for i, base_state in enumerate(base_state_list):
            for phi in phi_list:
                basis_dict[(i, phi)] = (rotation(phi) * base_state).unit()
        return basis_dict, phi_list

    def qudit_to_qubit(self, sigma_dit):
        d = self.d
        # this is the right way to do it
        sigma_dit.dims = [[2, d / 2], [2, d / 2]]
        return sigma_dit.ptrace(0)
        #
        # sigma_bit = np.zeros([2, 2])
        # for i, j in itertools.product(range(2), range(2)):
        #     sigma_bit[i, j] = sum([basis(d, int(i*d/2 + t)).dag() * sigma_dit * basis(d, int(j*d/2 + t)) for t in range(int(d/2))])[0][0][0]
        # return Qobj(sigma_bit).unit()


class EntangledBosonicQudit:
    def __init__(self, N, d1, res=1000, d2=None, base_state_list=None):
        """
        there isn't really support for d1 \neq d2.
        """
        self.N = N
        self.d1 = d1
        self.d2 = d1 if d2 is None else d2
        self.res = res
        BQ = BosonicQudit(N,d1,res=res)
        self.basis_dict, self.phi_list = BQ.create_basis_dictionary(self.res, base_state_list)

    def cavity_to_entangled_qudits(self, rho):
        """

        :param rho:
        :return:
        """
        d = self.d1
        # sigma = tensor(qeye(d), qeye(d))
        sigma = np.zeros([d*d, d*d])

        dphi = 2 * np.pi / d / self.res
        for i_A, i_B, j_A, j_B in itertools.product(range(d), range(d), range(d), range(d)):
            sigma[i_A * d + i_B, j_A * d + j_B] = dphi * sum([tensor(self.basis_dict[(i_A, phi_A)],
                                                                     self.basis_dict[(i_B, phi_B)]).dag()
                                      * rho *
                                      tensor(self.basis_dict[(j_A, phi_A)], self.basis_dict[(j_B, phi_B)])
                                      for phi_A, phi_B in itertools.product(self.phi_list, self.phi_list)])[0][0][0]
        return Qobj(sigma).unit()


        return sigma

class Qudit:
    def __init__(self,d):
        self.d = d

    @lru_cache(maxsize=10000000)
    def p_loss(self, gamma_loss: float, loss_times: int, alpha=0):
        """
        compute probability for loss.
        :param alpha:
        :param gamma_loss: the loss parameter
        :param loss_times: how many losses have been
        :return: p^(l=l)
        """
        if loss_times < 0 or loss_times >= self.d:
            return 0
        return gamma_loss**loss_times/np.math.factorial(loss_times) * (1-gamma_loss)**(alpha**2)
        # return stats.binom.pmf(l, self.d, gamma_loss)

    @lru_cache(maxsize=10000000)
    def p_dephasing(self, gamma_dephasing, s):
        if s > (self.d / 2):
            s = s - self.d
        return (erf(np.sqrt(1/(2 * gamma_dephasing)) * (2*s+1) * np.pi / self.d)
                - erf(np.sqrt(1/(2 * gamma_dephasing)) * (2*s-1) * np.pi / self.d))



class EntangledQudit:

    def __init__(self, d_A, d_B=None):
        self.d_A = d_A
        self.quditA = Qudit(d_A)
        self.d_B = d_B if d_B is not None else d_A
        self.quditB = Qudit(d_B)
        self.dit = lambda a, b: tensor(basis(self.d_A, a), basis(self.d_B, b))

    def qudit_from_list(self, digitList: (float, int, int)):
        '''
        :param digitList: a list of tuples in the form of [const,k_A,k_B].
        :return: normalized ket state of the entangled qudit
        '''
        return sum([tup[0]*self.dit(tup[1], tup[2]) for tup in digitList]).unit()

    @lru_cache(maxsize=10000000)
    def p(self, gamma_loss_A, gamma_dephasing_A, s_A,s_B,l_A,l_B, gamma_loss_B=None, gamma_dephasing_B=None):
        if gamma_loss_B is None:
            gamma_loss_B = gamma_loss_A
        if gamma_dephasing_B is None:
            gamma_dephasing_B = gamma_dephasing_A
        return (self.quditA.p_dephasing(gamma_dephasing_A, s_A) * self.quditB.p_dephasing(gamma_dephasing_B, s_B) *
                self.quditA.p_loss(gamma_loss_A, l_A) * self.quditB.p_loss(gamma_loss_B, l_B))

    # @lru_cache(maxsize=None)
    # @profile
    def fidelity_specific(self, A_1, A_2, B_1, B_2, m_i, m_c, gamma_loss_A, gamma_dephasing_A, m_f=2,
                          gamma_loss_B=None, gamma_dephasing_B=None, magic_state=False, no_com=False):
        """
        Calculates the fidelity for specific results A_1, A_2, B_1, B_2.
        We need to write all the possible dephasing errors and loss errors, and calculate the possible probabilities.
        :param A_1, B_1: Angular results. going from 0 to Delta_c-1
        :param A_2, B_2: photon-number results. going from 0 to m_c/2-1
        :param m_i:
        :param m_c:
        :param gamma_loss_A:
        :param gamma_dephasing_A:
        :param gamma_loss_B: NOT IMPLEMENTED AT THE MOMENT
        :param gamma_dephasing_B: NOT IMPLEMENTED AT THE MOMENT
        :return:
        """
        if gamma_loss_B is None:
            gamma_loss_B = gamma_loss_A
        if gamma_dephasing_B is None:
            gamma_dephasing_B = gamma_dephasing_A
        Delta_i = self.d_A / m_i
        Delta_c = self.d_A / m_c
        assert(0 <= A_1 < Delta_c)
        assert(0 <= B_1 < Delta_c)
        assert(0 <= A_2 < m_c/m_f)
        assert(0 <= B_2 < m_c/m_f)
        # all possible dephasing errors
        s_A_list = [A_1 + i * Delta_i for i in range(int(-m_i/2), int(m_i/2))]
        s_B_list = [B_1 + i * Delta_i for i in range(int(-m_i/2), int(m_i/2))]
        s_A_B_list = [(s_A, s_B) for s_A, s_B in itertools.product(s_A_list, s_B_list)
                      if ((s_A-s_B) % Delta_c) == ((A_1-B_1) % Delta_c)]
        if not no_com:
            # here we are searching for the best rotation for bob. it might be computationally intensive.
            dephasing_prob_tuple_list = [(u,
                                         sum([self.quditA.p_dephasing(gamma_dephasing_A, A_1 + t * Delta_i)
                                              * self.quditB.p_dephasing(gamma_dephasing_B,
                                                                        B_1 + t * Delta_i - u * Delta_c)
                                             for t in range(m_i)])
                                          ) for u in range(m_c)]
            u_B = max(dephasing_prob_tuple_list, key= lambda x: x[1])[0]
        else:
            x_A = A_1 if A_1 < (Delta_c-Delta_i/2) else A_1-Delta_c
            x_B = B_1 if B_1 < (Delta_c-Delta_i/2) else B_1-Delta_c
            u_B = (x_A - A_1 - x_B + B_1)/Delta_c
        good_s_A_B_list = [(s_A, s_B) for s_A, s_B in itertools.product(s_A_list, s_B_list)
                           if (s_A - s_B)%self.d_A == (A_1 - B_1 + u_B * Delta_c)%self.d_A]
        # this is the other way to do it, that has errors in it
        # good_s_A_B_list = [(s_A, s_B) for s_A, s_B in itertools.product(s_A_list, s_B_list)
        #                    if (s_A-s_B)%self.d_A == (min([(A_1-B_1)-i * Delta_c for i in range(-1, 2)], key=abs)
        #                                              % self.d_A)]

        # print(f"{s_A_B_list=}")
        # print(f"{good_s_A_B_list=}")

        # all possible loss errors
        l_A_list = list(range(self.d_A))
        l_B_list = list(range(self.d_B))

        l_A_B_list = [(l_A, l_B) for l_A, l_B in itertools.product(l_A_list, l_B_list)
                      if ((l_A+l_B) % int(m_c/m_f)) == (-(A_2+B_2) % int(m_c/m_f))]
        if not no_com:
            loss_prob_tuple_list = [(v,
                                     sum([self.quditA.p_loss(gamma_loss_A, m_c/2 - A_2 + t)
                                          * self.quditB.p_loss(gamma_loss_B, (1-v) * m_c / 2 - B_2 - t + j * m_c)
                                          for t, j in itertools.product(range(-int(m_c/2), int(m_c/2)),  # should I change this?
                                                                        range(-int(Delta_c), int(Delta_c)))])
                                     ) for v in range(m_f)]
            v_B = max(loss_prob_tuple_list, key=lambda x: x[1])[0]
        else:
            y_A = A_2
            y_B = B_2
            v_B = (y_A-y_B) / (m_c/2)  # guess I should change here
        # good_l_A_B_list = [(l_A, l_B) for l_A, l_B in l_A_B_list
        #                    if (l_A+l_B) == (-(A_2+B_2) % int(m_c/2))]
        good_l_A_B_list = [(l_A, l_B) for l_A, l_B in itertools.product(l_A_list, l_B_list)
                           if (l_A + l_B) % m_c == (- A_2 - B_2 + v_B * m_c / m_f) % m_c]

        if magic_state:
            l_A_list = [m_c/2 - A_2, m_c-A_2]
            l_B_list = [m_c/2 - B_2, m_c-B_2]
            l_A_B_list = [(l_A, l_B) for l_A, l_B in itertools.product(l_A_list, l_B_list)]
            good_l_A_B_list = [(m_c/2 - A_2, m_c/2 - B_2)]
        # print(f"{l_A_B_list=}")
        # print(f"{good_l_A_B_list=}")
        p_list = [self.p(gamma_loss_A=gamma_loss_A, gamma_dephasing_A=gamma_dephasing_A,
                         s_A=s_A, s_B=s_B, l_A=l_A, l_B=l_B)
                  for s_A, s_B in s_A_B_list
                  for l_A, l_B in l_A_B_list]
        good_p_list = [self.p(gamma_loss_A=gamma_loss_A, gamma_dephasing_A=gamma_dephasing_A,
                              s_A=s_A, s_B=s_B, l_A=l_A, l_B=l_B)
                       for s_A, s_B in good_s_A_B_list
                       for l_A, l_B in good_l_A_B_list]
        # p_dict = {f"{s_A}, {s_B}, {l_A}, {l_B}":
        #               self.p(gamma_loss_A=gamma_loss_A, gamma_dephasing_A = gamma_dephasing_A,
        #                      s_A=s_A, s_B=s_B, l_A=l_A, l_B=l_B)
        #           for s_A, s_B in s_A_B_list
        #           for l_A, l_B in l_A_B_list}
        # if not set(good_l_A_B_list).issubset(set(good_l_A_B_list2)):
        #     print("ahhhhaaa")
        return sum(good_p_list)/sum(p_list)
        # numerator = sum([])

    # @profile
    # @lru_cache(maxsize=None)
    def probability_specific(self, A_1, A_2, B_1, B_2, m_i, m_c, gamma_loss_A, gamma_dephasing_A, m_f=2,
                          gamma_loss_B=None, gamma_dephasing_B=None, magic_state=False):
        """
        Calculates the probability (not normalized) for specific results A_1, A_2, B_1, B_2.
        We need to write all the possible dephasing errors and loss errors, and calculate the possible probabilities.
        :param A_1, B_1: Angular results. going from 0 to Delta_c
        :param A_2, B_2: photon-number results. going from 0 to m_c/2-1
        :param m_i:
        :param m_c:
        :param gamma_loss_A:
        :param gamma_dephasing_A:
        :param gamma_loss_B:
        :param gamma_dephasing_B:
        :return:
        """
        Delta_i = self.d_A / m_i
        Delta_c = self.d_A / m_c
        s_A_list = [A_1 + i * Delta_i for i in range(m_i)]
        s_B_list = [B_1 + i * Delta_i for i in range(m_i)]
        s_A_B_list = [(s_A, s_B) for s_A, s_B in itertools.product(s_A_list, s_B_list)
                      if ((s_A-s_B) % Delta_c) == ((A_1-B_1) % Delta_c)]
        l_A_list = list(range(self.d_A))
        l_B_list = list(range(self.d_B))
        l_A_B_list = [(l_A, l_B) for l_A, l_B in itertools.product(l_A_list, l_B_list)
                      if ((l_A+l_B) % int(m_c/m_f)) == (-(A_2+B_2) % int(m_c/m_f))]
        if magic_state:
            l_A_list = [m_c/2 - A_2, m_c-A_2]
            l_B_list = [m_c/2 - B_2, m_c-B_2]
            l_A_B_list = [(l_A, l_B) for l_A, l_B in itertools.product(l_A_list, l_B_list)]
            good_l_A_B_list = [(m_c/2 - A_2, m_c/2 - B_2)]
        p_list = [self.p(gamma_loss_A=gamma_loss_A, gamma_dephasing_A=gamma_dephasing_A,
                         s_A=s_A, s_B=s_B, l_A=l_A, l_B=l_B)
                  for s_A, s_B in s_A_B_list
                  for l_A, l_B in l_A_B_list]
        return sum(p_list)

    def probability_list(self, gamma_loss_A, gamma_dephasing_A):
        return [self.p(gamma_loss_A, gamma_dephasing_A, s1, s2, l1, l2) for s1, s2, l1, l2 in
                itertools.product(range(int(self.d_A)), range(int(self.d_B)),
                                  range(int(self.d_A)), range(int(self.d_B)))]

    def fidelity_trivial(self, m_i, s_A, s_B, l_A, l_B):
        """
        Returns the fidelity for the trivial case (no protocol)
        :param s_A, s_B, l_A, l_B: The different errors.
        :param m_i: Initial state rotation order
        :param gamma_loss_A:
        :param gamma_dephasing_A:
        :param gamma_loss_B:
        :param gamma_dephasing_B:
        :return:
        """
        Delta_i = self.d_A / m_i
        # Create the initial state
        initial = sum([np.exp(2j * np.pi * (l_A+l_B) * Delta_i * k / self.d_A) *
                       tensor(basis(self.d_A, int((Delta_i * (k+1/2)+s_A) % self.d_A)),
                              basis(self.d_B, int((Delta_i * (k+1/2)+s_B) % self.d_B)))
                       for k in range(m_i)])
        initial.dims = [[2, int(self.d_A/2), 2, int(self.d_A/2)] , [1,1,1,1]]
        # do a "vacuum cleaner"
        traced_state = initial.ptrace([0, 2]).unit()
        # print(traced_state)
        # compare to a bell state
        bell_state = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
        return (fidelity(traced_state, ket2dm(bell_state).unit()))**2

    def transform_to_fourier_basis(self, qudit: Qobj, reverse=False):
        plus_basis_list1 = []
        plus_basis_list2 = []
        direction = 1 if reverse else -1
        for base_number in range(self.d_A):
            plus_basis_list1.append(sum([np.exp(direction * 2j * np.pi * i * base_number / self.d_A)
                                         * basis(self.d_A, i) for i in range(self.d_A)]))
        for base_number in range(self.d_B):
            plus_basis_list2.append(sum([np.exp(direction * 2j * np.pi * i * base_number / self.d_B)
                                         * basis(self.d_B, i) for i in range(self.d_B)]))
        tensor_plus_basis_list = [tensor(alice_plus,bob_plus).unit() for alice_plus, bob_plus in itertools.product(plus_basis_list1,plus_basis_list2)]
        return qudit.transform(tensor_plus_basis_list)

    def print_qudit(self, qudit):
        string_list = []
        for (i, val) in enumerate(np.array(qudit)):
            if val[0] == 0:
                continue
            index = np.unravel_index(i, shape=(self.d_A, self.d_B))
            string_list.append("+ \\left|" + str(index[0]) + "," + str(index[1]) + "\\right\\rangle")
        return ''.join(string_list)

    # def p_loss_dephasing_both(self,l_A,l_B,s_A,s_B, gamma_loss_A, gamma_dephasing_A,
    #                           gamma_loss_B = None, gamma_dephasing_B = None):
    #     """
    #
    #     :param l_A:
    #     :param l_B:
    #     :param s_A:
    #     :param s_B:
    #     :param gamma_loss_A:
    #     :param gamma_dephasing_A:
    #     :param gamma_loss_B:
    #     :param gamma_dephasing_B:
    #     :return:
    #     """




# N = 60
# d = 4
# a = 4
# qudit = BosonicQudit(N, d)
# sigma = qudit.cavity_to_qudit(ket2dm(coherent(N, a)+coherent(N, -a)).unit())