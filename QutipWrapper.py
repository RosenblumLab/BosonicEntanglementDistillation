import qutip


class QutipWrapper(qutip.Qobj):

    creation_functions = {}

    @classmethod
    def create(cls, name: str, number_of_fock_states: int, *args, **kwargs):
        return cls(cls.creation_functions[name](number_of_fock_states, *args, **kwargs))
