import numpy as np
from scipy.interpolate import interp1d
import os
import sys
sys.path.append('../')
import save_sklearn_gp as ssg

from model import model_base

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
        
        self.angles = [0, 30, 45, 60, 75, 90]

        interpolator_suffixes = ["%03d" % i for i in range(191)]

        self.t_interp_full = np.logspace(np.log10(0.125), np.log10(7.6608262), 191)

        self.interpolators = {angle:[] for angle in self.angles}

        ### rather than preload all the interpolators, just store their string names and load them on the fly
        for i, suffix in enumerate(interpolator_suffixes):
            for angle in self.angles:
                if angle == 0: angle = '00'
                self.interpolators[int(angle)].append(interp_loc + "theta" + str(angle) + "deg/t_" + "{:1.3f}".format(self.t_interp_full[int(suffix)]) + "_days/model")
        
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
        
        self.params_array = None # internal storage of parameter array
        self.theta = None # internal storage of theta specifically (for convenience)

    def model_predict(self, model, inputs):

        from scipy.linalg import cholesky, cho_solve

        K = model.kernel_(model.X_train_)
        K[np.diag_indices_from(K)] += model.alpha
        model.L_ = cholesky(K, lower=True) # recalculating L matrix since this is what makes the pickled models bulky
        model._K_inv = None # has to be set to None so the GP knows to re-calculate matrices used for uncertainty
        K_trans = model.kernel_(inputs, model.X_train_)
        pred = K_trans.dot(model.alpha_)
        pred = model._y_train_std * pred + model._y_train_mean
        v = cho_solve((model.L_, True), K_trans.T)
        y_cov = model.kernel_(inputs) - K_trans.dot(v)
        err = np.sqrt(np.diag(y_cov))
        return pred, err

    def set_params(self, params, t_bounds):
        ### params should be a dictionary mapping parameter names to either single floats or 1d arrays.
        ### if it's a float, convert it to an array
        if isinstance(params["mej_dyn"], float):
            self.params_array = np.empty((1, 5))
            self.theta = np.array([params["theta"]])
        else:
            self.params_array = np.empty((params["mej_dyn"].size, 5))
            self.theta = params["theta"]

        ### make a dictionary mapping angular bins - e.g. (0, 30) - to arrays of integers.
        ### these arrays give the indices of self.params_array with theta values inside that angular bin.
        self.index_dict = {}
        for angle_index in range(len(self.angles) - 1):
            theta_lower = self.angles[angle_index]
            theta_upper = self.angles[angle_index + 1]
            self.index_dict[(theta_lower, theta_upper)] = np.where((theta_lower <= self.theta) & (self.theta < theta_upper))[0]
        
        ### now populate the parameter array
        self.params_array[:,0] = params["mej_dyn"]
        self.params_array[:,1] = params["vej_dyn"]
        self.params_array[:,2] = params["mej_wind"]
        self.params_array[:,3] = params["vej_wind"]


    def evaluate(self, tvec_days, band):
        print(band + " band:")
        self.params_array[:,4] = self.lmbda_dict[band]

        ### find out which interpolators we actually need to use
        ind_list = [] # list of interpolator indices (i.e. an integer 0, 1, ..., 190) that are used
        t_interp = [] # list of times corresponding to these interpolators
        for t in tvec_days:
            for i in range(190):
                if self.t_interp_full[i] <= t < self.t_interp_full[i + 1]:
                    if i not in ind_list:
                        t_interp.append(self.t_interp_full[i])
                        ind_list.append(i)
                    if i + 1 not in ind_list:
                        t_interp.append(self.t_interp_full[i + 1])
                        ind_list.append(i + 1)

        t_interp = np.array(t_interp)

        ### 2d arrays to hold the interpolator values.
        ### each row is one light curve corresponding to the parameter values in that row of self.params_array.
        ### each column is a time value corresponding to t_interp
        mags_interp = np.empty((self.params_array.shape[0], t_interp.size))
        mags_err_interp = np.empty((self.params_array.shape[0], t_interp.size))

        for lc_index in range(t_interp.size):
            if lc_index == 0 or (lc_index + 1) % 5 == 0:
                print("  evaluating time step {} of {}".format(lc_index + 1, t_interp.size))
            interp_index = ind_list[lc_index]
            
            ### cache the GP objects to avoid reloading one we already have
            GP_dict = {}
            
            ### iterate over angular bins
            for angle_index in range(len(self.angles) - 1):
                theta_lower = self.angles[angle_index]
                theta_upper = self.angles[angle_index + 1]
                delta_theta = float(theta_upper) - float(theta_lower)
                param_indices = self.index_dict[(theta_lower, theta_upper)] # indices of self.params_array corresponding to this angular bin
                if param_indices.size == 0: # skip loading and evaluating the interpolators if we have no points to evaluate
                    continue
                if theta_lower in GP_dict.keys(): # check if we've already loaded this interpolator to avoid extra load time
                    interp_lower = GP_dict[theta_lower]
                else:
                    interp_lower = ssg.load_gp(self.interpolators[theta_lower][interp_index])
                interp_upper = ssg.load_gp(self.interpolators[theta_upper][interp_index]) # this interpolator will never already be loaded so no need to check
                GP_dict[theta_upper] = interp_upper # cache the upper interpolator (we won't need the lower one again)
                
                ### evaluate the interpolator at this time step for the upper and lower angles
                mags_lower, mags_err_lower = self.model_predict(interp_lower, self.params_array[param_indices])
                mags_upper, mags_err_upper = self.model_predict(interp_upper, self.params_array[param_indices])

                ### insert these values in the column of mags_interp corresponding to this time step and the row(s) corresponding to this angular bin
                mags_interp[:,lc_index][param_indices] = ((theta_upper - self.theta[param_indices]) * mags_lower
                        + (self.theta[param_indices] - theta_lower) * mags_upper) / delta_theta
                mags_err_interp[:,lc_index][param_indices] = ((theta_upper - self.theta[param_indices]) * mags_err_lower
                        + (self.theta[param_indices] - theta_lower) * mags_err_upper) / delta_theta
                
        ### now we need to construct the light curves at the user-requested times
        ### start by creating empty arrays with rows corresponding to rows of self.params_array and columns corresponding to the user-requested times
        mags_out = np.empty((self.params_array.shape[0], tvec_days.size))
        mags_err_out = np.empty((self.params_array.shape[0], tvec_days.size))
        
        ### iterate over light curves (or parameter combinations, depending on how you look at it)
        for i in range(self.params_array.shape[0]):
            ### make a 1d interpolator for magnitudes and another for errors
            mags_interpolator = interp1d(t_interp, mags_interp[i], fill_value="extrapolate")
            mags_err_interpolator = interp1d(t_interp, mags_err_interp[i], fill_value="extrapolate")
            ### evaluate
            mags_out[i] = mags_interpolator(tvec_days)
            mags_err_out[i] = mags_err_interpolator(tvec_days)
        
        if self.params_array.shape[0] == 1:
            ### if the model is being used in non-vectorized form, return 1d arrays
            return mags_out.flatten(), mags_err_out.flatten()
        return mags_out, mags_err_out
