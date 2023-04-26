import qutip
import matplotlib.pyplot as plt

from Simulation import Simulation

if __name__ == '__main__':
   simulation = Simulation(
      number_of_fock_states=30,
      number_of_rotations=4,
      number_of_parties=1,
      initial_state_name='pegg-barnett',
      number_of_parity_sectors=4)
   qutip.plot_wigner(simulation.state_after_protocol)
   plt.show()

