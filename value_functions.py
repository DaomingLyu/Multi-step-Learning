import numpy as np

class BairdValueFunction():
    ### construct the feature matrix for baird domain ##
    def __init__(self):
        self.param_shape = (2*8, )
        phis = np.array([
            [2., 0., 0., 0., 0., 0., 0., 1.],
            [0., 2., 0., 0., 0., 0., 0., 1.],
            [0., 0., 2., 0., 0., 0., 0., 1.],
            [0., 0., 0., 2., 0., 0., 0., 1.],
            [0., 0., 0., 0., 2., 0., 0., 1.],
            [0., 0., 0., 0., 0., 2., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 2.]
        ])

        PHI = np.zeros((7, 2, self.param_shape[0]))
        for s in range(7):
            PHI[s, 0, :] = np.concatenate([phis[s], np.zeros(phis.shape[1])])
            PHI[s, 1, :] = np.concatenate([np.zeros(phis.shape[1]), phis[s]])
        self.PHI = PHI

    def feature(self, s, a):
        return self.PHI[s, a, :]


class CounterExampleValueFunction():
    ### construct the feature matrix for the two-state MDP domain ##
    def __init__(self):
        self.param_shape = (2, )
        self.PHI = np.zeros((2, 2, 2))
        self.PHI[0, 0, :] = np.array([1., 0.])
        self.PHI[0, 1, :] = np.array([0., 1.])
        self.PHI[1, 0, :] = np.array([2., 0.])
        self.PHI[1, 1, :] = np.array([0., 2.])

    def feature(self, s, a):
        return self.PHI[s, a, :]


