import abc
from typing import Tuple, Union, List
import qutip as qt
import numpy as np
from scipy.special import gammaln, factorial, erf, erfc

# from rotation_code_utils import factorial_stirling_root as fs

"""
This file was created by Peter Leviant as part of arXiv:2205.00341v4 
"""

class POVM(abc.ABC):
    """
    This class represents a POVM - sum(M_x)=I
    The operator are assumed of equal weight, so any expression int(dxp(x)M_x)=I should be changed to
    int(dx M_x^')=I with appropriate weighting M_x^'=M_x*p(x)
    Effect operators are A_x s.t. M_x=A_x*A_x^dag and a density rho goes to A_x*rho*A_x^dagger after
    measuring x
    """

    def __init__(self):
        self.POVM_elements = {
            x: self._operator_for_result(x) for x in self.get_possible_results()
        }

    @abc.abstractmethod
    def get_possible_results(self) -> List[Union[Tuple[float, ...], float]]:
        """
        :return: The possible results [x...]
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _operator_for_result(self, x: Union[Tuple[float, ...], float]) -> qt.Qobj:
        """
        :param x: the possible result
        :return: M_x
        """
        raise NotImplementedError()

    def get_prob_for_result(self, x: Union[Tuple[float, ...], float], state: qt.Qobj) -> float:
        """
        the probability for the result x, given a state
        :param x: the possible result
        :param state: the state we measure
        :return:
        """
        return qt.expect(self.POVM_elements[x], state)

    @abc.abstractmethod
    def effect_for_result(self, x: Union[Tuple[float, ...], float]) -> qt.Qobj:
        """
        The effect super operator that happens after measuring x
        :param x: the possible result
        :return: E_x*E_x^dag
        """
        raise NotImplementedError()


class CanonicalPhaseMeasurement(POVM):
    def __init__(self, base_state: qt.Qobj, result_resolution: int, num: qt.Qobj):
        """
        theta_n = 2pi*n/resolution,
        M_n=M(theta_n)=1/resolution*sum(gamma)gamma_m*(gamma_n^*)*e^(i*theta(m-n))|m><n|
        with gamma_n = <n|state>/|<n|state>| if |<n|state>|!=0 and 1 otherwise
        :param base_state: the state that generates the canonical measurement
        :param result_resolution: resolution of phase measurement - it will be from 0,2pi/resolution,
        to 2pi*(resolution-1)/resolution
        :param num: number operator for the state
        """
        self.num = num
        self.number_normalized_state, self._coefficients = self.get_number_normalized_state_from_state(base_state)
        self.result_resolution = result_resolution
        super().__init__()

    @staticmethod
    def get_number_normalized_state_from_state(state: qt.Qobj) -> Tuple[qt.Qobj, List[float]]:
        """
        modified_state = sum(gamma_i|i>) gamma_i=<i|state>/|<i|state>| if <i|state>!=0 and 1 otherwise
        :param state: state
        :return: modified_state
        """
        N = state.shape[0]
        new_coefficients = [qt.basis(N, i).overlap(state) for i in range(N)]
        new_coefficients = [new_coefficients[i] / np.abs(new_coefficients[i]) if new_coefficients[i] != 0 else 1 for i
                            in range(N)]
        return sum([new_coefficients[i] * qt.basis(N, i) for i in range(N)]), new_coefficients

    def get_possible_results(self) -> List[Union[Tuple[float, ...], float, int]]:
        """
        possible results are angles
        :return: list of angles
        """
        return [2 * np.pi * i / self.result_resolution for i in range(self.result_resolution)]

    def _operator_for_result(self, x: float) -> qt.Qobj:
        """
        integrating e^(i n phi)|base><base|e^(-i n phi) over interval
        :param x: middle of resulting interval (x-np.pi/res,x+np.pi/res)
        :return: POVM operator
        """
        phi_plus = x + np.pi / self.result_resolution
        phi_minus = x - np.pi / self.result_resolution
        N = self.num.shape[0]
        gamma = self._coefficients
        return 1 / (2 * np.pi) * qt.Qobj(np.array([[
            (gamma[m] * gamma[n].conjugate() * (np.exp(1j * (m - n) * phi_plus) - np.exp(1j * (m - n) * phi_minus)) / (
                    1j * (m - n)) if m != n else (phi_plus - phi_minus)) for n in range(N)] for m in range(N)]))

    def effect_for_result(self, x: float) -> qt.Qobj:
        raise NotImplementedError()


class IdealHeterodynePhaseMeasurement(POVM):
    def __init__(self, phi_resolution: int, num: qt.Qobj):
        """
        This measurement is M_phi = 1/pi*int rdr|r*e^iphi><r*e^iphi|.
        phi = i*2pi/phi_resolution, r=max_r*(j+1)/r_resolution for i in range(phi_resolution) and j in
        range(r_resolution)
        :param phi_resolution: resolution of phase measurement - it will be from 0,2pi/resolution,
        to 2pi*(resolution-1)/resolution
        :param num: number operator for the state, gives range offset,...,offset+N
        """
        self.num = num
        self.phi_resolution = phi_resolution
        self.ray_projection = self.get_ray_projection()
        super().__init__()

    def get_ray_projection(self) -> qt.Qobj:
        """
        generates the state int rdr |r><r|. Integral is calculated analytically
        :return: ray state int rdt |r><r|
        """
        return qt.Qobj(self.ray_coeff()) / np.pi

    def ray_coeff(self):
        offset = int(self.num.data[0, 0])
        N = self.num.shape[0]
        n = np.arange(offset, N + offset)
        m = np.arange(offset, N + offset)
        n, m = np.meshgrid(n, m)
        even = ((n + m + 1) % 2) * (
                1 / 2 * np.exp(1 / 2 * (self.binomial_log(n + m, n) - self.binomial_log(n + m, (n + m) / 2))))
        odd = ((n + m) % 2) * (np.pi * (n + m + 1)) ** 0.5 * np.exp(1 / 2 * (
                self.binomial_log(n + m, n) + self.binomial_log(n + m + 1, (n + m + 1) / 2) - (
                2 * (n + m + 1) + 2) * np.log(2)))
        return odd + even

    @staticmethod
    def binomial_log(n, k):
        return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)

    def get_possible_results(self) -> List[Union[Tuple[float, ...], float]]:
        return [2 * np.pi * i / self.phi_resolution for i in range(self.phi_resolution)]

    def _operator_for_result(self, x: float) -> qt.Qobj:
        phi_plus = x + np.pi / self.phi_resolution
        phi_minus = x - np.pi / self.phi_resolution
        N = self.num.shape[0]
        n = np.arange(N)
        m = np.arange(N)
        n, m = np.meshgrid(n, m)
        coeff = (np.exp(1j * (m - n) * phi_plus) - np.exp(1j * (m - n) * phi_minus)) / (1j * (m - n) + (m == n)) + (
                phi_plus - phi_minus) * (m == n)
        return qt.Qobj(np.multiply(coeff, self.ray_projection.full()))

    def effect_for_result(self, x: float) -> qt.Qobj:
        raise NotImplementedError()


class HeterodynePhaseMeasurement(POVM):
    def __init__(self, phi_resolution: int, num: qt.Qobj, eta: float, max_r=12, steps=300):
        """
        This measurement is M_phi = 1/pi*int rD(r)Thermal(eta)D(r)^dag dr
        eta = 1/(1+mean_number_of_photons)
        phi = i*2pi/phi_resolution, r=max_r*(j+1)/r_resolution for i in range(phi_resolution) and j in
        range(r_resolution)
        :param phi_resolution: resolution of phase measurement - it will be from 0,2pi/resolution,
        to 2pi*(resolution-1)/resolution
        :param num: number operator for the state, gives range offset,...,offset+N
        :param eta: efficiency
        """
        self.num = num
        self.eta = eta
        self.max_r = max_r
        self.steps = steps
        self.phi_resolution = phi_resolution
        self.xvec = np.linspace(-self.max_r, self.max_r, 2 * self.steps + 1)
        self.q, self.p = np.meshgrid(self.xvec, self.xvec, indexing="xy")
        self.r = np.sqrt(self.q ** 2 + self.p ** 2)
        self.r[self.steps, self.steps] = self.r[self.steps, self.steps + 1] / 1000
        self.phi = np.angle(self.q + 1j * self.p)
        self.phi_exp = np.exp(1j * self.phi)
        self.r_exp = np.exp(- self.r ** 2 / 2)
        self.ray_projection = self.get_ray_projection()
        self.ray_projection = self.ray_projection / (2 * np.pi * max(self.ray_projection.diag()))
        super().__init__()

    def get_ray_projection(self) -> qt.Qobj:
        """
        generates the state int rdr |r><r|. Integral is calculated analytically
        :return: ray state int rdt |r><r|
        """
        step_size = self.max_r / self.steps
        P = self.get_P_ray_for_efficiency()
        return qt.Qobj(np.array(
            [[2 * np.pi * np.multiply(self.get_Q_number_projector(j, i), P).sum() * step_size ** 2 for i in
              range(self.num.shape[0])]
             for j in
             range(self.num.shape[0])]))

    def get_thermal_P_func(self):
        """
        get P function using P(q,p)=1/(2*pi*N)exp(-(q^2+p^2)/2N) where eta=1/(1+N)
        and rho_eta=int P(q,p)|(q+ip)/sqrt(2)><(q+ip)/sqrt(2)|dqdp
        :param eta: efficiency
        :return:
        """
        N = 1 / self.eta - 1
        xvec = np.linspace(-self.max_r, self.max_r, 2 * self.steps + 1)
        q, p = np.meshgrid(xvec, xvec, indexing="xy")
        return 1 / (2 * np.pi * N) * np.exp(-(q ** 2 + p ** 2) / (2 * N))

    def get_P_ray_for_efficiency(self):
        """
        Adding shifts of the P functions
        :param eta: efficiency
        :return:
        """
        N = 1 / self.eta - 1
        return 1 / (4 * np.pi * N) * np.exp(-self.r ** 2 / (2 * N)) * (
                    2 * N + np.exp(self.q ** 2 / (2 * N)) * np.sqrt(2 * np.pi * N) * self.q *(1+ erf(
                self.q / np.sqrt(2 * N))))
        #
        # P = self.get_thermal_P_func()
        # step_size = self.max_r / self.steps
        # rolling_matrix = np.asmatrix(
        #     np.array([[max(i - j + 1, 0) for i in range(2 * self.steps + 1)] for j in range(2 * self.steps + 1)]))
        # return np.asmatrix(P) * rolling_matrix * step_size ** 2 / (2 * np.pi)

    def get_Q_number_projector(self, n, m):
        """
        Q function of |n><m|
        :param max_r:
        :param steps:
        :return:
        """
        print(n, m)
        return 1 / (2 * np.pi) * np.exp(-1 / 2 * (gammaln(n + 1) + gammaln(m + 1)) + (n + m) * np.log(
            self.r / (2 ** 0.5))) * self.r_exp * self.phi_exp ** (m - n)

    @staticmethod
    def binomial_log(n, k):
        return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)

    def get_possible_results(self) -> List[Union[Tuple[float, ...], float]]:
        return [2 * np.pi * i / self.phi_resolution for i in range(self.phi_resolution)]

    def _operator_for_result(self, x: float) -> qt.Qobj:
        phi_plus = x + np.pi / self.phi_resolution
        phi_minus = x - np.pi / self.phi_resolution
        N = self.num.shape[0]
        n = np.arange(N)
        m = np.arange(N)
        n, m = np.meshgrid(n, m)
        coeff = (np.exp(1j * (m - n) * phi_plus) - np.exp(1j * (m - n) * phi_minus)) / (1j * (m - n) + (m == n)) + (
                phi_plus - phi_minus) * (m == n)
        return qt.Qobj(np.multiply(coeff, self.ray_projection.full()))

    def effect_for_result(self, x: float) -> qt.Qobj:
        raise NotImplementedError()


class PrettyGoodMeasurement(POVM):
    """
    A pretty good measurement! given a noise channel, and initial encoding of d states,
    this determines which of the encoded states we are looking at
    """

    def __init__(self, noise_channel: qt.Qobj, encoded_states):
        """
        M_i = sigma^(-1/2)N(|i><i|)sigma^(-1/2), sigma=N(sum(|i><i|))
        :param noise_channel: a super operator which determines the measurement
        :param encoded_states: the encoded states are orthogonal and denoted by |phi_0>...|phi_d-1>
        """
        self.noise_channel = noise_channel
        self.encoded_states = encoded_states
        self.sigma = self.get_sigma()
        self.one_over_root_sigma = self.one_over_root(self.sigma)
        super().__init__()

    def get_possible_results(self, include_complement: bool = False) -> List[float]:
        """
        The possible results here are 0,1,...,d-1 or -1 which is I-Projector over the states
        :param include_complement: should this include I-Projector or not
        :return: [-1,0,1,2,3,...,d]
        """
        if include_complement:
            return list(range(-1, len(self.encoded_states)))
        else:
            return list(range(len(self.encoded_states)))

    def _operator_for_result(self, x: int) -> qt.Qobj:
        """
        :param x: between -1 and d-1
        :return: I-P_sigma if x=-1 and sigma^(-1/2)*N(|x><x|)*sigma^(-1/2) otherwise
        """
        if x == -1:
            return self.one_over_root_sigma * self.sigma * self.one_over_root_sigma
        else:
            return self.one_over_root_sigma * self.noise_channel(
                self.encoded_states[x].proj()) * self.one_over_root_sigma

    def effect_for_result(self, x: Union[Tuple[float, ...], float]) -> qt.Qobj:
        raise NotImplementedError()

    def get_sigma(self) -> qt.Qobj:
        """
        sigma=N(sum(|i><i|).
        :return: sigma
        """
        P = sum([state.proj() for state in self.encoded_states])
        sigma = self.noise_channel(P)
        return sigma

    @staticmethod
    def one_over_root(operator):
        """
        given an operator, it caluclates the square of its inverse in its support
        :param operator:
        :return: (operator|_support)^(-1/2)
        """
        eigenstates = operator.eigenstates()[1]
        diagonal_operator = operator.transform(eigenstates)
        return qt.Qobj(
            np.diag(np.array([1 / np.sqrt(x) if x > 0 else 0 for x in list(diagonal_operator.diag())]))).transform(
            eigenstates, inverse=True)
