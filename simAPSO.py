import numpy as np
import pandas as pd



def simrandom(size, vlim=None, scale=True):

    if isinstance(size, float):
        size = int(size)

    if isinstance(size, (tuple, list)):
        size = np.array(size).astype(int)

    if vlim is None:
        return np.random.random(size)
    else:
        if scale:
            return np.random.random(size) * (vlim[1] - vlim[0]) + vlim[0]
        else:
            return np.random.random(size) * (vlim[1] - vlim[0])


class SIMAPSO(object):
    def __init__(self, objective_func,
                 n_particles,
                 dimensions,
                 c1=2.05, c2=2.05, lamda=0.98, bounds=None, init_pos=None):
        """
            :param objective_func: 待优化函数
            :param n_particles: 粒子数目
            :param dimensions: 自变量个数
            :param c1: 学习因子1
            :param c2: 学习因子2
            :param lamda: 退火常数
            :param init_pos: 初始值
            :param bounds:自变量的取值范围
            :return:
                    xm:目标函数取最小值时的自变量值
                    fv：目标函数的最小值
        """

        self.objective_func = objective_func
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.miters = None
        self.c1 = c1
        self.c2 = c2
        self.lamda = lamda
        self.bounds = bounds
        self.init_pos = init_pos

        self.x_pos = np.random.random((self.n_particles, self.dimensions))
        self.v_pos = np.random.random((self.n_particles, self.dimensions))
        self.y_pos = self.x_pos.copy()
        self.piv = np.zeros(self.n_particles)

        self.gpos = self.x_pos[-1, :]  # save the global best
        self.gposp = self.x_pos[np.random.randint(0, len(self.x_pos)), :]
        self.gfv = np.random.randint(99999)
        self.T = None
        self.pcost = []
        self.gcost = np.inf
        self.cost_history = []

        # initialization
        self.initialize()

    def initialize(self):
        # init position and velocity
        if self.bounds is not None:
            self.x_pos = np.zeros((self.n_particles, self.dimensions))
            self.v_pos = np.zeros((self.n_particles, self.dimensions))
            for ic, xlim in enumerate(zip(self.bounds[0], self.bounds[1])):
                tmxi = simrandom(self.n_particles, vlim=xlim, scale=True)
                self.x_pos[:, ic] = tmxi

                tmvi = simrandom(self.n_particles, vlim=xlim, scale=False)
                self.v_pos[:, ic] = tmvi

        # set initial position
        if self.init_pos is not None:
            self.x_pos[np.random.randint(0, len(self.x_pos)), :] = self.init_pos
        else:
            pass

        # compute function values
        for ipt in range(self.n_particles):
            self.piv[ipt] = self.objective_func(self.x_pos[ipt])

        for ipt in range(self.n_particles - 1):
            tmfit1 = self.objective_func(self.x_pos[ipt])
            tmfit2 = self.objective_func(self.gpos)
            if tmfit1 < tmfit2:
                self.gpos = self.x_pos[ipt, :]
                self.gfv = tmfit1
            else:
                self.gfv = tmfit2
        self.T = self.objective_func(self.gpos) / np.log(5)  # init temperature
        # self.T = self.objective_func(self.gpos)

    def bounds_protect(self):
        for ic, xlim in enumerate(zip(self.bounds[0], self.bounds[1])):
            tmidx = self.x_pos[:, ic] > xlim[1]
            self.x_pos[tmidx, ic] = xlim[1] - np.random.random(1)[0] * (xlim[1] - xlim[0])

            tmidx = self.x_pos[:, ic] < xlim[0]
            self.x_pos[tmidx, ic] = xlim[0] + np.random.random(1)[0] * (xlim[1] - xlim[0])

    def random_pgp(self):
        # best fit
        groupFit = self.objective_func(self.gpos)
        # each Pi values at current temperature
        # print('piv:', self.piv)
        # print('gbest:',groupFit )
        # print('T:',self.T)
        # print('4:',(self.piv - groupFit) / self.T)
        fitIndex = np.exp(-(self.piv - groupFit) / self.T)  # 0224改
        # fitIndex = -(self.piv - groupFit) / self.T  # 计算每一个粒子的适应度
        # print(fitIndex)
        Tfit = fitIndex / np.nansum(fitIndex)
        # 用轮盘赌策略确定全局最优的某个替代值
        # p_best = np.abs(np.random.randn(1)[0])
        p_best = np.random.rand()
        # print('best:',p_best)
        # print('Tfit:',Tfit)
        ComFit = np.zeros(self.n_particles)
        for ipt in range(self.n_particles):
            ComFit[ipt] = np.nansum(Tfit[:ipt+1])
            if p_best <= ComFit[ipt]:
                self.gposp = self.x_pos[ipt, :]
                print(ipt, p_best, ComFit[ipt])
                break
            else:
                pass

    def optimize(self, miters=100):
        self.miters = miters
        for ite in range(self.miters):
            # random pg
            # self.gposp = self.x_pos[np.random.randint(0, len(self.x_pos)), :]
            self.random_pgp()
            # update velocity and position
            C = self.c1 + self.c2
            ksi = 2 / np.abs(2 - C - np.sqrt(C ** 2 - 4 * C))  # compress factor
            tmp_cost = []
            for ipt in range(self.n_particles):
                rand1, rand2 = np.random.random(1)[0], np.random.random(1)[0]
                self.v_pos[ipt, :] = ksi * (self.v_pos[ipt, :] +
                                            self.c1 * rand1 * (self.y_pos[ipt, :] - self.x_pos[ipt, :]) +
                                            self.c2 * rand2 * (self.gposp - self.x_pos[ipt, :]))
                self.x_pos[ipt, :] = self.x_pos[ipt, :] + self.v_pos[ipt, :]

                if self.bounds is not None:
                    self.bounds_protect()

                tmfit = self.objective_func(self.x_pos[ipt, :])
                tmp_cost.append(tmfit)
                if tmfit < self.piv[ipt]:
                    self.piv[ipt] = tmfit
                    self.y_pos[ipt, :] = self.x_pos[ipt, :]

                tmfit = self.objective_func(self.gpos)

                if self.piv[ipt] < tmfit:
                    print('>>>', ipt)
                    self.gpos = self.y_pos[ipt, :]
                    # print(self.gpos)
                    self.gfv = tmfit
            # print('piv:', self.piv)
            self.pcost.append(tmp_cost)
            self.cost_history.append(np.nanmin(self.pcost[ite]))

            # update temperature
            # self.T = np.max([self.T * self.lamda, 0.001])
            self.T *= self.lamda

    def plothistory(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.plot(self.cost_history, 'k-*')
        plt.show()


def fitness(x):
    F = 0
    for i in range(len(x)):
        F = F + 1 / (i + 1 + (x[i] - 1)**2)
    return 1 / (0.01 + F)


def holdertable(x):
    """Holder Table objective function

    Only takes two dimensions and has a four equal global minimums
     of `-19.2085` at :code:`f([8.05502, 9.66459])`, :code:`f([-8.05502, 9.66459])`,
     :code:`f([8.05502, -9.66459])`, and :code:`f([-8.05502, -9.66459])`.
    Its coordinates are bounded within :code:`[-10, 10]`.

    Best visualized with the full domain and a range of :code:`[-20, 0]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """

    # if not x.shape[1] == 2:
    #     raise IndexError(
    #         "Holder Table function only takes two-dimensional input."
    #     )
    # if not np.logical_and(x >= -10, x <= 10).all():
    #     raise ValueError(
    #         "Input for Holder Table function must be within [-10,10]."
    #     )

    if not np.logical_and(x >= -10, x <= 10).all():
        return 99999999999.0

    if len(x.shape) == 2:
        x_ = x[:, 0]
        y_ = x[:, 1]
    else:
        x_ = x[0]
        y_ = x[1]

    j = -np.abs(
        np.sin(x_)
        * np.cos(y_)
        * np.exp(np.abs(1 - np.sqrt(x_ ** 2 + y_ ** 2) / np.pi))
    )

    return j


if __name__ == '__main__':
    simapso = SIMAPSO(holdertable, n_particles=20, dimensions=2, c1=2.05, c2=2.05, bounds=([-10, -10], [10, 10]))
    simapso.optimize(miters=100)
    # print(simapso.gfv, simapso.gcost)
    print(simapso.gfv, simapso.gpos)
    simapso.plothistory()



