import qutip
import numpy as np

from QutipWrapper import QutipWrapper


class State(QutipWrapper):
    @staticmethod
    def __basis(number_of_fock_states: int, *args, **kwargs):
        return qutip.basis(number_of_fock_states, *args, **kwargs)

    @staticmethod
    def __coherent(number_of_fock_states: int, *args, **kwargs):
        return qutip.coherent(number_of_fock_states, *args, **kwargs)

    @staticmethod
    def __create_pegg_barnett_state(number_of_fock_states: int, angle_in_radians: float = 0):
        state = qutip.zero_ket(number_of_fock_states)

        for i in range(number_of_fock_states):
            state += np.exp(1j * i * angle_in_radians) * qutip.basis(number_of_fock_states, i)

        return state

    creation_functions = {
        'basis': __basis,
        'coherent': __coherent,
        'pegg-barnett': __create_pegg_barnett_state,
    }

    @classmethod
    def create(cls, name: str, number_of_fock_states: int, *args, **kwargs):
        return cls(super().create(name, number_of_fock_states, *args, **kwargs).unit())
