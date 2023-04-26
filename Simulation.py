from State import State
from Operator import Operator


class Simulation:
    def __init__(
            self,
            number_of_fock_states: int = 80,
            number_of_rotations: int = 4,
            number_of_parties: int = 1,
            initial_state_name: str = 'pegg-barnett',
            number_of_parity_sectors: int = 4,
            *args,
            **kwargs):

        self.number_of_fock_states = number_of_fock_states
        self.number_of_rotations = number_of_rotations
        self.number_of_parties = number_of_parties
        self.initial_state_name = initial_state_name
        self.number_of_parity_sectors = number_of_parity_sectors

        self.__initialize_states(*args, **kwargs)

    def __initialize_states(self, *args, **kwargs):
        self.initial_state = self.__create_initial_state(*args, **kwargs)
        self.noisy_state = self.initial_state
        self.__add_noise()
        self.state_after_protocol = self.__run_protocol()

    def __create_initial_state(self, *args, **kwargs) -> State:
        return State.create(
            name=self.initial_state_name,
            number_of_fock_states=self.number_of_fock_states,
            number_of_parties=self.number_of_parties,
            number_of_rotations=self.number_of_rotations,
            rotate_before_sum=False,
            *args,
            **kwargs)

    def __add_noise(self):
        pass

    def __run_protocol(self) -> State:
        wigner_parity_operator = Operator.create(
            name='wigner-parity',
            number_of_fock_states=self.number_of_fock_states,
            number_of_rotated_parts=self.number_of_parity_sectors,
            number_of_parties=self.number_of_parties)

        return self.noisy_state.evaluate_operator(wigner_parity_operator)
