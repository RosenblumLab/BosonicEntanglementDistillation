import itertools
from qutip import *
import numpy as np
from qudit import *


class DiscreteSimulation:
    def __init__(self, d, m_i, m_c, magic=False):
        self.d = d
        self.m_i = m_i
        self.m_c = m_c
        self.magic = magic
        self.enQudit = EntangledQudit(d, d)

    def average_fidelity(self, gamma_loss, gamma_dephasing, fidelity_cut=0, **kwargs):
        fidelity_sum = 0
        prob_sum = 0
        good_prob = 0
        # going on all possible combination and counting them
        for A_1, B_1, A_2, B_2 in itertools.product(range(int(self.d / self.m_c)), range(int(self.d / self.m_c)),
                                                    range(int(self.m_c / 2)),
                                                    range(int(self.m_c / 2))):  # , total = int(d/2*d/2)):
            fid = self.enQudit.fidelity_specific(gamma_loss_A=gamma_loss, gamma_dephasing_A=gamma_dephasing, A_1=A_1,
                                                 B_1=B_1, A_2=A_2,
                                                 B_2=B_2, m_i=self.m_i, m_c=self.m_c, magic_state=self.magic, **kwargs)
            prob = self.enQudit.probability_specific(gamma_loss_A=gamma_loss, gamma_dephasing_A=gamma_dephasing,
                                                     A_1=A_1, B_1=B_1,
                                                     A_2=A_2, B_2=B_2, m_i=self.m_i, m_c=self.m_c,
                                                     magic_state=self.magic)
            # print(A_1,B_1,A_2,B_2)
            if prob != 0:
                if fid >= fidelity_cut:
                    fidelity_sum += prob * fid
                    good_prob += prob
                prob_sum += prob
        if good_prob > 0:
            average_fid = fidelity_sum / good_prob
        else:
            average_fid = 0
        fail_probability = (prob_sum - good_prob) / prob_sum
        return average_fid, fail_probability

    def average_fidelity_no_communication(self, gamma_loss, gamma_dephasing):
        return self.average_fidelity(gamma_loss, gamma_dephasing, fidelity_cut=0, no_com=True)

    def probability_sum(self, gamma_loss, gamma_dephasing):
        """
        Notice that this function gives back p_total,including the division by m_i/m_f because of the collapse.
        :param gamma_loss: gamma loss
        :param gamma_dephasing: gamma dephasing
        :return:
        """
        prob_sum = 0
        for A_1, B_1, A_2, B_2 in itertools.product(range(int(self.d / self.m_c)), range(int(self.d / self.m_c)),
                                                    range(int(self.m_c / 2)),
                                                    range(int(self.m_c / 2))):  # , total = int(d/2*d/2)):
            prob = self.enQudit.probability_specific(gamma_loss_A=gamma_loss, gamma_dephasing_A=gamma_dephasing,
                                                     A_1=A_1, B_1=B_1,
                                                     A_2=A_2, B_2=B_2, m_i=self.m_i, m_c=self.m_c,
                                                     magic_state=self.magic)
            prob_sum += prob
        return prob_sum / (self.m_i/2)


    # def average_fidelity_local_filter(self, gamma_loss, gamma_dephasing, fidelity_cut=0, prob_cut=0):
    #     fidelity_sum = 0
    #     prob_sum = 0
    #     good_prob = 0
    #     # going on all possible combination and counting them
    #     for A_1, B_1, A_2, B_2 in itertools.product(range(int(self.d / self.m_c)), range(int(self.d / self.m_c)),
    #                                                 range(int(self.m_c / 2)),
    #                                                 range(int(self.m_c / 2))):  # , total = int(d/2*d/2)):
    #         fid = self.enQudit.fidelity_specific(gamma_loss_A=gamma_loss, gamma_dephasing_A=gamma_dephasing, A_1=A_1,
    #                                              B_1=B_1, A_2=A_2,
    #                                              B_2=B_2, m_i=self.m_i, m_c=self.m_c, magic_state=self.magic)
    #         prob = self.enQudit.probability_specific(gamma_loss_A=gamma_loss, gamma_dephasing_A=gamma_dephasing,
    #                                                  A_1=A_1, B_1=B_1,
    #                                                  A_2=A_2, B_2=B_2, m_i=self.m_i, m_c=self.m_c,
    #                                                  magic_state=self.magic)
    #         if prob != 0:
    #             print(f"{A_1}")
    #             if fid >= fidelity_cut:
    #                 fidelity_sum += prob * fid
    #                 good_prob += prob
    #             prob_sum += prob
    #     if good_prob > 0:
    #         average_fid = fidelity_sum / good_prob
    #     else:
    #         average_fid = 0
    #     fail_probability = (prob_sum - good_prob) / prob_sum
    #     return average_fid, fail_probability
