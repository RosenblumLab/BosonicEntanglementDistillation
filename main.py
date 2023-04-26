import numpy as np
import qutip

from State import State
from Operator import Operator

if __name__ == '__main__':
   pegg_barnett = State.create("pegg-barnett", 30)
   Operator = Operator.create("rotation", 30, np.pi)
   print(qutip.fidelity(State.create("pegg-barnett", 30, angle_in_radians=np.pi), Operator * pegg_barnett))
