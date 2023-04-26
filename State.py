import qutip
import numpy as np

from Operator import Operator
from QutipWrapper import QutipWrapper


class State(QutipWrapper):
    @staticmethod
    def __create_basis(number_of_fock_states: int, *args, **kwargs):
        return qutip.basis(number_of_fock_states, *args, **kwargs)

    @staticmethod
    def __create_coherent(number_of_fock_states: int, *args, **kwargs):
        return qutip.coherent(number_of_fock_states, *args, **kwargs)

    @staticmethod
    def __create_pegg_barnett(number_of_fock_states: int, angle_in_radians: float = 0):
        state = qutip.zero_ket(number_of_fock_states)

        for i in range(number_of_fock_states):
            state += np.exp(1j * i * angle_in_radians) * qutip.basis(number_of_fock_states, i)

        return state

    creation_functions = {
        'basis': __create_basis,
        'coherent': __create_coherent,
        'pegg-barnett': __create_pegg_barnett,
    }

    @classmethod
    def create(
            cls,
            name: str,
            number_of_fock_states: int,
            number_of_parties: int = 1,
            number_of_rotations: int = 1,
            rotate_before_sum: bool = True,
            *args,
            **kwargs):
        return cls(super().create(
            name,
            number_of_fock_states,
            number_of_parties,
            number_of_rotations,
            rotate_before_sum,
            *args,
            **kwargs).unit())

    def apply_operator(self, operator: qutip.Qobj):
        if self.dims[1][0] == 1:
            return self.__class__(operator * self)
        return self.__class__(super().apply_operator(operator))

    def evaluate_operator(self, operator: Operator):
        return self.apply_operator(operator).unit()
