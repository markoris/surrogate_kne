import numpy as np
from scipy.interpolate import interp1d
from joblib import load
import os

from .model import model_base

class kn_interp_angle(model_base):
    def __init__(self):
        name = "kn_interp_angle"
        param_names = ["mej_dyn", "vej_dyn", "mej_wind", "vej_wind", "theta"]
        bands = ["g", "r", "i", "z", "y", "J", "H", "K"]
        model_base.__init__(self, name, param_names, bands)
        self.vectorized = True
        
        interp_loc = os.environ["INTERP_LOC"]
        if interp_loc[-1] != "/":
            interp_loc += "/"
        #full_times = np.loadtxt(interp_loc + "times.dat")
        self.angles = [0, 30, 60, 75, 90]

        #ind_use = np.arange(191) % 1 == 0 # use every interpolator
    
        interpolator_suffixes = ["%03d" % i for i in range(191)]

        # force it to always use first and last interpolators
        #ind_use[0] = True
        #ind_use[-1] = True

        #self.t_interp = full_times[ind_use]
        self.t_interp_full = np.loadtxt(interp_loc + "times.dat")

        self.interpolators = {angle:[] for angle in self.angles}

        #for i, suffix in enumerate(interpolator_suffixes):
        #    if not ind_use[i]:
        #        continue
        #    for angle in self.angles:
        #        print("loading interpolator", suffix, "angle", angle)
        #        self.interpolators[angle].append(load(interp_loc + "saved_models_angle/ang" + str(angle) + "_time_" + suffix + ".joblib"))
        
        ### rather than preload all the interpolators, just store their string names and load them on the fly
        for i, suffix in enumerate(interpolator_suffixes):
            #if not ind_use[i]:
            #    continue
            for angle in self.angles:
                self.interpolators[angle].append(interp_loc + "saved_models_angle/ang" + str(angle) + "_time_" + suffix + ".joblib")
        
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
        self.ind_0_30 = None
        self.ind_30_60 = None
        self.ind_60_75 = None
        self.ind_75_90 = None
        self.theta = None

    def set_params(self, params, t_bounds):
        if isinstance(params["mej_dyn"], float):
            self.params_array = np.empty((1, 5))
            self.theta = np.array([params["theta"]])
        else:
            self.params_array = np.empty((params["mej_dyn"].size, 5))
            self.theta = params["theta"]

        ### find indices corresponding to each angular bin
        self.ind_0_30 = np.where(self.theta < 30)[0]
        self.ind_30_60 = np.where((30 <= self.theta) & (self.theta < 60))[0]
        self.ind_60_75 = np.where((60 <= self.theta) & (self.theta < 75))[0]
        self.ind_75_90 = np.where(75 <= self.theta)[0]
        
        self.params_array[:,0] = params["mej_dyn"]
        self.params_array[:,1] = params["vej_dyn"]
        self.params_array[:,2] = params["mej_wind"]
        self.params_array[:,3] = params["vej_wind"]


    def evaluate(self, tvec_days, band):
        print(band + " band:")
        mags_out = np.empty((self.params_array.shape[0], tvec_days.size))
        mags_err_out = np.empty((self.params_array.shape[0], tvec_days.size))
        self.params_array[:,4] = self.lmbda_dict[band]

        ### find out which interpolators we actually need to use
        ind_use = [False] * 191
        t_interp = []
        for t in tvec_days:
            for i in range(190):
                if self.t_interp_full[i] <= t < self.t_interp_full[i + 1]:
                    if not ind_use[i]:
                        ind_use[i] = True
                        t_interp.append(self.t_interp_full[i])
                    if not ind_use[i + 1]:
                        ind_use[i + 1] = True
                        t_interp.append(self.t_interp_full[i + 1])

        t_interp = np.array(t_interp)
        mags_interp = np.empty((self.params_array.shape[0], t_interp.size))
        mags_err_interp = np.empty((self.params_array.shape[0], t_interp.size))
        
        for i in range(t_interp.size):
            if i == 0 or (i + 1) % 5 == 0:
                print("  evaluating time step {} of {}".format(i + 1, t_interp.size))
            ### 0-30 angular bin
            interpolator_0 = load(self.interpolators[0][i])
            interpolator_30 = load(self.interpolators[30][i])
            if self.ind_0_30.size > 0:
                mags_0, mags_err_0 = interpolator_0.GP.evaluate(self.params_array[self.ind_0_30])
                mags_30, mags_err_30 = interpolator_30.GP.evaluate(self.params_array[self.ind_0_30])
                mags_0 *= interpolator_0.std
                mags_0 += interpolator_0.mean
                mags_err_0 *= interpolator_0.std
                mags_30 *= interpolator_30.std
                mags_30 += interpolator_30.mean
                mags_err_30 *= interpolator_30.std
                mags_interp[:,i][self.ind_0_30] = ((30.0 - self.theta[self.ind_0_30]) * mags_0 + (self.theta[self.ind_0_30] - 0.0) * mags_30) / (30.0)
                mags_err_interp[:,i][self.ind_0_30] = ((30.0 - self.theta[self.ind_0_30]) * mags_err_0 + (self.theta[self.ind_0_30] - 0.0) * mags_err_30) / (30.0)
            
            ### 30-60 angular bin
            interpolator_60 = load(self.interpolators[60][i])
            if self.ind_30_60.size > 0:
                mags_30, mags_err_30 = interpolator_30.GP.evaluate(self.params_array[self.ind_30_60])
                mags_60, mags_err_60 = interpolator_60.GP.evaluate(self.params_array[self.ind_30_60])
                mags_30 *= interpolator_30.std
                mags_30 += interpolator_30.mean
                mags_err_30 *= interpolator_30.std
                mags_60 *= interpolator_60.std
                mags_60 += interpolator_60.mean
                mags_err_60 *= interpolator_60.std
                mags_interp[:,i][self.ind_30_60] = ((60.0 - self.theta[self.ind_30_60]) * mags_30 + (self.theta[self.ind_30_60] - 30.0) * mags_60) / (30.0)
                mags_err_interp[:,i][self.ind_30_60] = ((60.0 - self.theta[self.ind_30_60]) * mags_err_30 + (self.theta[self.ind_30_60] - 30.0) * mags_err_60) / (30.0)

            ### 60-75 angular bin
            interpolator_75 = load(self.interpolators[75][i])
            if self.ind_60_75.size > 0:
                mags_60, mags_err_60 = interpolator_60.GP.evaluate(self.params_array[self.ind_60_75])
                mags_75, mags_err_75 = interpolator_75.GP.evaluate(self.params_array[self.ind_60_75])
                mags_60 *= interpolator_60.std
                mags_60 += interpolator_60.mean
                mags_err_60 *= interpolator_60.std
                mags_75 *= interpolator_75.std
                mags_75 += interpolator_75.mean
                mags_err_75 *= interpolator_75.std
                mags_interp[:,i][self.ind_60_75] = ((75.0 - self.theta[self.ind_60_75]) * mags_60 + (self.theta[self.ind_60_75] - 60.0) * mags_75) / (15.0)
                mags_err_interp[:,i][self.ind_60_75] = ((75.0 - self.theta[self.ind_60_75]) * mags_err_60 + (self.theta[self.ind_60_75] - 60.0) * mags_err_75) / (15.0)

            ### 75-90 angular bin
            interpolator_90 = load(self.interpolators[90][i])
            if self.ind_75_90.size > 0:
                mags_75, mags_err_75 = interpolator_75.GP.evaluate(self.params_array[self.ind_75_90])
                mags_90, mags_err_90 = interpolator_90.GP.evaluate(self.params_array[self.ind_75_90])
                mags_75 *= interpolator_75.std
                mags_75 += interpolator_75.mean
                mags_err_75 *= interpolator_75.std
                mags_90 *= interpolator_90.std
                mags_90 += interpolator_90.mean
                mags_err_90 *= interpolator_90.std
                mags_interp[:,i][self.ind_75_90] = ((90.0 - self.theta[self.ind_75_90]) * mags_75 + (self.theta[self.ind_75_90] - 75.0) * mags_90) / (15.0)
                mags_err_interp[:,i][self.ind_75_90] = ((90.0 - self.theta[self.ind_75_90]) * mags_err_75 + (self.theta[self.ind_75_90] - 75.0) * mags_err_90) / (15.0)

        for i in range(self.params_array.shape[0]):
            mags_interpolator = interp1d(t_interp, mags_interp[i], fill_value="extrapolate")
            mags_err_interpolator = interp1d(t_interp, mags_err_interp[i], fill_value="extrapolate")
            mags_out[i] = mags_interpolator(tvec_days)
            mags_err_out[i] = mags_err_interpolator(tvec_days)
        
        if self.params_array.shape[0] == 1:
            # if the model is being used in non-vectorized form, return 1d arrays
            return mags_out.flatten(), mags_err_out.flatten()
        return mags_out, mags_err_out
