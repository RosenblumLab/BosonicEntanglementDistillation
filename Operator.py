import qutip
import numpy as np

from itertools import product

from QutipWrapper import QutipWrapper


class Operator(QutipWrapper):

    @staticmethod
    def _create_number(number_of_fock_states: int):
        return qutip.num(number_of_fock_states)

    @staticmethod
    def _create_destroy(number_of_fock_states: int):
        return qutip.destroy(number_of_fock_states)

    @staticmethod
    def _create_angle_projection(
            number_of_fock_states: int,
            number_of_parts: int,
            row: int = 0,
            col: int = 0,
            center_angle_in_radians: float = 0):
        operator = qutip.qzero(number_of_fock_states)

        for n, m in product(range(number_of_fock_states), repeat=2):
            coefficient = Operator._get_angle_projection_coefficient(
                n,
                m,
                number_of_parts,
                row,
                col,
                center_angle_in_radians)
            matrix_element = qutip.basis(number_of_fock_states, n) * qutip.basis(number_of_fock_states, m).dag()
            operator += coefficient * matrix_element

        return operator

    @staticmethod
    def _get_angle_projection_coefficient(
            n: int,
            m: int,
            number_of_parts: int,
            row: int,
            col: int,
            center_angle_in_radians: float):
        coefficient = 0.5 * np.exp(2 * np.pi * n * (col - row) * 1j / number_of_parts)
        if n != m:
            coefficient = 1 / (n - m) * np.sin(np.pi * (n - m) / number_of_parts) * \
                          np.exp(2 * np.pi * 1j * (n * col - m * row) / number_of_parts
                                 + (n - m) * center_angle_in_radians * 1j) / np.pi
        return coefficient

    @staticmethod
    def _create_coherent_projection(number_of_fock_states: int, *args, **kwargs):
        return qutip.ket2dm(qutip.coherent(number_of_fock_states, *args, **kwargs))

    @staticmethod
    def _create_rotation(number_of_fock_states: int, angle_in_radians: float):
        number_operator = qutip.num(number_of_fock_states)
        return (angle_in_radians * 1j * number_operator).expm()

    @staticmethod
    def _create_wigner_parity(number_of_fock_states: int, number_of_rotated_parts: int):
        operator = qutip.qzero(number_of_fock_states)

        for i in range(0, number_of_rotated_parts, 2):
            operator += Operator.create(
                'angle projection',
                number_of_fock_states=number_of_fock_states,
                number_of_parts=number_of_rotated_parts,
                center_angle_in_radians=2 * np.pi * i / number_of_rotated_parts)

        return operator

    @staticmethod
    def _create_fock_parity(number_of_fock_states: int, number_of_parties: int):
        operator = QutipWrapper.repeat(number_of_fock_states, number_of_parties)

        for i in range(0, number_of_fock_states, 2):
            operator += QutipWrapper.repeat(qutip.ket2dm(qutip.basis(number_of_fock_states, i)), number_of_parties)

        return operator

    creation_functions = {
        'number': _create_number,
        'destroy': _create_destroy,
        'angle projection': _create_angle_projection,
        'coherent projection': _create_coherent_projection,
        'rotation': _create_rotation,
        'wigner-parity': _create_wigner_parity,
        'fock-parity': _create_fock_parity
    }
