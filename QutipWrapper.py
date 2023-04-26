import qutip

import numpy as np


class QutipWrapper(qutip.Qobj):

    creation_functions = {}

    @classmethod
    def __create_rotated(
            cls,
            base,
            number_of_fock_states: int,
            number_of_parties: int = 1,
            number_of_rotations: int = 1):

        number_operator = qutip.num(number_of_fock_states)
        state = base

        for i in range(1, number_of_rotations):
            rotated_base = cls.apply_operator(
                base,
                (2 * np.pi * i * 1j / number_of_rotations * number_operator).expm())
            state += qutip.tensor(*(rotated_base,) * number_of_parties)
        return state

    @classmethod
    def __create_local(cls, base, number_of_fock_states: int, number_of_parties: int = 1, number_of_rotations: int = 1):
        state = cls.__create_rotated(base, number_of_fock_states, 1, number_of_rotations)
        return qutip.tensor(*(state,) * number_of_parties)

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

        base = cls.creation_functions[name](number_of_fock_states, *args, **kwargs)

        if rotate_before_sum:
            return cls.__create_local(base, number_of_fock_states, number_of_parties, number_of_rotations)
        return cls.__create_rotated(base, number_of_fock_states, number_of_parties, number_of_rotations)

    def apply_operator(self, operator: qutip.Qobj):
        return self.__class__(operator * self * operator.inv())

