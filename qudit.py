import itertools
from qutip import *
import numpy as np


class BosonicQudit:
    def __init__(self, N, d, res = 1000):
        self.N = N
        self.d = d
        self.res = res
        self.proj_op_list = self.create_proj_list()
        self.basis_dict, self.phi_list = self.create_basis_dictionary(self.res)

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

    def create_basis_dictionary(self, res):
        """
        creates a dictionary that gets a tuple (i,phi) and returns the relevant vector
        :res: the resolution in which we work at
        :return: Dictionary of kets.
        """
        N = self.N
        d = self.d
        pegg_bernet_0 = 1 / np.sqrt(N) * sum([basis(N, i) for i in range(N)])
        rotation = lambda x: (x * 1.0j * num(N)).expm()
        phi_minus = -np.pi/d
        phi_plus = np.pi/d
        phi_list = np.linspace(phi_minus, phi_plus, res)
        basis_dict = {}
        for i in range(d):
            for phi in phi_list:
                basis_dict[(i, phi)] = (rotation(2 * i * np.pi / d) * rotation(phi) * pegg_bernet_0).unit()
        return basis_dict, phi_list


class EntangledBosonicQudit:
    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        None

    def cavity_to_entangled_qudits(self, rho):
        """

        :param rho:
        :return:
        """
        d1 = self.d1
        d2 = self.d2
        sigma = tensor(qeye(d1), qeye(d2))


        return sigma


# N = 60
# d = 4
# a = 4
# qudit = BosonicQudit(N, d)
# sigma = qudit.cavity_to_qudit(ket2dm(coherent(N, a)+coherent(N, -a)).unit())