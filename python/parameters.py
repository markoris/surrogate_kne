import numpy as np
from scipy.stats import loguniform

class Parameter:
    def __init__(self, name, llim, rlim):
        self.name = name
        self.llim = llim
        self.rlim = rlim

    def update_limits(self, llim, rlim):
        self.llim = llim
        self.rlim = rlim

class UniformPriorParameter(Parameter):
    def __init__(self, name, llim, rlim):
        Parameter.__init__(self, name, llim, rlim)

    def prior(self, x):
        return 1.0 / (self.rlim - self.llim)

    def sample_from_prior(self, size=1, width=1.0):
        d = (self.rlim - self.llim) * (1.0 - width) / 2.0
        ret = np.random.uniform(self.llim + d, self.rlim - d, size=size)
        return ret if size != 1 else ret[0]


class LogUniformPriorParameter(Parameter):
    def __init__(self, name, llim, rlim):
        Parameter.__init__(self, name, llim, rlim)

    def prior(self, x):
        return loguniform.pdf(x, self.llim, self.rlim) # FIXME should I be setting the loc and scale for this?

    def sample_from_prior(self, size=1, width=1.0):
        ret = loguniform.rvs(*loguniform.interval(width, self.llim, self.rlim), size=size)
        return ret if size != 1 else ret[0]


class Distance(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "dist", 10.0, 100.0)


class EjectaMass(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej", 0.001, 0.1)


class EjectaMassRed(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej_red", 0.001, 0.1)


class EjectaMassPurple(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej_purple", 0.001, 0.1)


class EjectaMassBlue(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej_blue", 0.001, 0.1)


class DynamicalEjectaMass(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej_dyn", 0.001, 0.1)


class WindEjectaMass(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "mej_wind", 0.001, 0.1)


class EjectaVelocity(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej", 0.1, 0.4)


class EjectaVelocityRed(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej_red", 0.1, 0.4)


class EjectaVelocityPurple(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej_purple", 0.1, 0.4)


class DynamicalEjectaVelocity(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej_dyn", 0.05, 0.3)


class WindEjectaVelocity(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej_wind", 0.05, 0.3)


class EjectaVelocityBlue(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "vej_blue", 0.1, 0.4)


class TcRed(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "Tc_red", 3500.0, 4000.0)


class TcPurple(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "Tc_purple", 1000.0, 1500.0)


class TcBlue(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "Tc_blue", 400.0, 1000.0)


class Kappa(LogUniformPriorParameter):
    def __init__(self):
        LogUniformPriorParameter.__init__(self, "kappa", 0.1, 10.0)

class Sigma(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "sigma", 0.0, 0.5)

class Theta(UniformPriorParameter):
    def __init__(self):
        UniformPriorParameter.__init__(self, "theta", 0.0, 90.0)
