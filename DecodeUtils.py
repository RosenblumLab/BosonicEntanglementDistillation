import qutip
import itertools
import numpy as np

from State import State
from Operator import Operator
from QutipWrapper import QutipWrapper


class DecodeUtils:
    @staticmethod
    def _create_operator_list(
            rows: tuple[int],
            cols: tuple[int],
            number_of_fock_states: int,
            number_of_parts: int,
            center_angle_in_radians: float = 0) -> list[Operator]:
        operator_list = []

        for (row, col) in zip(rows, cols):
            sub_operator = Operator.create('angle-projection',
                                           number_of_fock_states=number_of_fock_states,
                                           number_of_parts=number_of_parts,
                                           row=row,
                                           col=col,
                                           center_angle_in_radians=center_angle_in_radians)
            operator_list.append(sub_operator)

        return operator_list

    @staticmethod
    def decode(
            state: State,
            number_of_fock_states: int,
            number_of_parts: int,
            number_of_parties: int,
            center_angle_in_radians: float = 0) -> qutip.Qobj:
        decoded_matrix = np.zeros(
            [number_of_parts ** number_of_parties, number_of_parts ** number_of_parties],
            dtype=complex)

        for rows in itertools.product(range(number_of_parts), repeat=number_of_parties):
            for cols in itertools.product(range(number_of_parts), repeat=number_of_parties):
                operator_list = DecodeUtils.__create_operator_list(
                    rows,
                    cols,
                    number_of_fock_states,
                    number_of_parts,
                    center_angle_in_radians)
                operator = QutipWrapper.tensor(operator_list)

                decoded_matrix[np.ravel_multi_index(rows, [number_of_parts] * number_of_parties),
                np.ravel_multi_index(cols, [number_of_parts] * number_of_parties)] = np.trace(state * operator)

        return qutip.Qobj(decoded_matrix)
