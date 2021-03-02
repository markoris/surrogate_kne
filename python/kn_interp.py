import numpy as np
from scipy.interpolate import interp1d
from joblib import load
import os

from .model import model_base

class kn_interp(model_base):
    def __init__(self):
        name = "kn_interp"
        param_names = ["mej_dyn", "vej_dyn", "mej_wind", "vej_wind"]
        bands = ["g", "r", "i", "z", "y", "J", "H", "K"]
        model_base.__init__(self, name, param_names, bands)
        self.vectorized = True
        
        interp_loc = os.environ["INTERP_LOC"]
        if interp_loc[-1] != "/":
            interp_loc += "/"
        full_times = np.loadtxt(interp_loc + "times.dat")

        ind_use = np.arange(191) % 4 == 0 # use every 4th interpolator
    
        interpolator_suffixes = ["%03d" % i for i in range(191)]

        # force it to always use first and last interpolators
        ind_use[0] = True
        ind_use[-1] = True

        self.t_interp = full_times[ind_use]

        self.interpolators = []

        for i, suffix in enumerate(interpolator_suffixes):
            if not ind_use[i]:
                continue
            print("loading interpolator", suffix)
            self.interpolators.append(load(interp_loc + "saved_models/time_" + suffix + ".joblib"))
        
        self.lmbda_dict = { # dictionary of wavelengths corresponding to bands
                "u":354.3,
                "g":477.56,
                "r":612.95,
                "i":748.46,
                "z":865.78,
                "y":960.31,
                "J":1235.0,
                "H":1662.0,
                "K":2159.0
        }

        self.params_array = None

    def set_params(self, params, t_bounds):
        if isinstance(params["mej_dyn"], float):
            self.params_array = np.empty((1, 5))
        else:
            self.params_array = np.empty((params["mej_dyn"].size, 5))
        self.params_array[:,0] = params["mej_dyn"]
        self.params_array[:,1] = params["vej_dyn"]
        self.params_array[:,2] = params["mej_wind"]
        self.params_array[:,3] = params["vej_wind"]

    def evaluate(self, tvec_days, band):
        mags_out = np.empty((self.params_array.shape[0], tvec_days.size))
        mags_err_out = np.empty((self.params_array.shape[0], tvec_days.size))
        self.params_array[:,4] = self.lmbda_dict[band]

        mags_interp = np.empty((self.params_array.shape[0], self.t_interp.size))
        mags_err_interp = np.empty((self.params_array.shape[0], self.t_interp.size))
        
        for i, interpolator in enumerate(self.interpolators):
            mags_interp[:,i], mags_err_interp[:,i] = interpolator.GP.evaluate(self.params_array)
            mags_interp[:,i] *= interpolator.std
            mags_interp[:,i] += interpolator.mean
            mags_err_interp[:,i] *= interpolator.std

        for i in range(self.params_array.shape[0]):
            mags_interpolator = interp1d(self.t_interp, mags_interp[i], fill_value="extrapolate")
            mags_err_interpolator = interp1d(self.t_interp, mags_err_interp[i], fill_value="extrapolate")
            mags_out[i] = mags_interpolator(tvec_days)
            mags_err_out[i] = mags_err_interpolator(tvec_days)
        
        if self.params_array.shape[0] == 1:
            # if the model is being used in non-vectorized form, return 1d arrays
            return mags_out.flatten(), mags_err_out.flatten()
        return mags_out, mags_err_out
