from .interpolated_model import *
from .kilonova import *
#from .kilonova_3c import *
from .kn_interp import *
from .kn_interp_angle import *

from .parameters import *

model_dict = {
        "interpolated":interpolated,
        "kilonova":kilonova,
        "kilonova_3c":kilonova_3c,
        "kn_interp":kn_interp,
        "kn_interp_angle":kn_interp_angle
}

param_dict = {
        "dist":Distance,
        "mej":EjectaMass,
        "mej_red":EjectaMassRed,
        "mej_purple":EjectaMassPurple,
        "mej_blue":EjectaMassBlue,
        "mej_dyn":DynamicalEjectaMass,
        "mej_wind":WindEjectaMass,
        "vej":EjectaVelocity,
        "vej_red":EjectaVelocityRed,
        "vej_purple":EjectaVelocityPurple,
        "vej_blue":EjectaVelocityBlue,
        "vej_dyn":DynamicalEjectaVelocity,
        "vej_wind":WindEjectaVelocity,
        "Tc_red":TcRed,
        "Tc_purple":TcPurple,
        "Tc_blue":TcBlue,
        "kappa":Kappa,
        "sigma":Sigma,
        "theta":Theta
}
