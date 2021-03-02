# -*- coding: utf-8 -*-
"""
Interpolated Model
------------------
Model interpolated from some surrogate model
"""
import numpy as np
import os
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor

from .model import model_base

class interpolated(model_base):
    def __init__(self, name, param_names, bands, weight=1):
        model_base.__init__(self, name, param_names, bands, weight)
        fname = os.environ["EM_PE_INSTALL_DIR"] + "/Data/" + self.name + ".npz"
        f = np.load(fname)
        self.t_interp = f["arr_0"]
        x = f["arr_1"]
        lc_arr = f["arr_2"]
        self.gp_dict = {}
        for i in range(len(self.bands)):
            band = self.bands[i]
            y = lc_arr[i]
            self.gp_dict[band] = []
            for j in range(len(self.t_interp)):
                gp = GaussianProcessRegressor()
                gp.fit(x, y[:,j])
                self.gp_dict[band].append(gp)

    def evaluate(self, tvec_days, band):
        x = np.empty((1, len(self.param_names)))
        gp_list = self.gp_dict[band]
        for i in range(len(self.param_names)):
            x[0][i] = self.params[self.param_names[i]]
        y = np.empty(len(self.t_interp))
        for i in range(len(self.t_interp)):
            gp = gp_list[i]
            y[i] = gp.predict(x)
        f = interpolate.interp1d(self.t_interp, y, fill_value="extrapolate")
        return f(tvec_days), 0
