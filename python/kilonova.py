import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumtrapz

from .model import model_base

class kilonova(model_base):
    def __init__(self):
        name = "kilonova"
        param_names = ["mej",
                       "vej",
                       "kappa",
                       "sigma"]
        bands = ["g", "r", "i", "z", "y", "J", "H", "K"]
        model_base.__init__(self, name, param_names, bands)
        
        _v = np.array([0.1, 0.2, 0.3])
        _m = np.array([1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2])
        _a = np.asarray([[2.01, 4.52, 8.16], [0.81, 1.9, 3.2],
                         [0.56, 1.31, 2.19], [.27, .55, .95]])
        _b = np.asarray([[0.28, 0.62, 1.19], [0.19, 0.28, 0.45],
                         [0.17, 0.21, 0.31], [0.10, 0.13, 0.15]])
        _d = np.asarray([[1.12, 1.39, 1.52], [0.86, 1.21, 1.39],
                         [0.74, 1.13, 1.32], [0.6, 0.9, 1.13]])
        self.fa = RegularGridInterpolator((_m, _v), _a, bounds_error=False, fill_value=None)
        self.fb = RegularGridInterpolator((_m, _v), _b, bounds_error=False, fill_value=None)
        self.fd = RegularGridInterpolator((_m, _v), _d, bounds_error=False, fill_value=None)

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

        self.tdays = None
        self.R_photo = None
        self.T_photo = None
    
    def set_params(self, params, t_bounds):
        """
        Method to set the parameters for lightcurve
        model.

        Parameters
        ----------
        params : dict
            Dictionary mapping parameter names to their values
        t_bounds : list
            [upper bound, lower bound] pair for time values
        """
        self.sigma = params["sigma"]
        ### conversion factors and physical constants
        Msun = 1.988409870698051e33 # g
        c = 2.99792458e10 # cm/s
        sigmaSB = 5.67e-5 # erg cm^-2 s^-1 K^-4
        d2s = 24.0 * 3600.0

        ### initialize time arrays
        tmin = 1.0e-6
        tmax = t_bounds[1]
        n = 1000
        tdays = np.logspace(np.log10(tmin), np.log10(tmax), n)
        self.tdays = tdays
        t = tdays * d2s

        ### constants
        t0 = 1.3 # s
        sigma = 0.11 # s
        beta = 13.7
        Tc = 4000.0 # K

        ### compute photosphere radius and temperature
        mej, vej, kappa = params["mej"], params["vej"], params["kappa"]
        a = self.fa([mej, vej])[0]
        b = self.fb([mej, vej])[0]
        d = self.fd([mej, vej])[0]
        vej *= c
        td = np.sqrt(2.0 * kappa * (mej * Msun) / (beta * vej * c))
        L_in = 4.0e18 * (mej * Msun) * (0.5 - np.arctan((t - t0) / sigma) / np.pi)**1.3
        e_th = 0.36 * (np.exp(-a * tdays) + np.log1p(2.0 * b * tdays**d) / (2.0 * b * tdays**d))
        L_in *= e_th

        integrand = L_in * t * np.exp((t / td)**2) / td
        L_bol = np.empty(t.size)
        L_bol[1:] = cumtrapz(integrand, t)
        L_bol[0] = L_bol[1]
        L_bol *= 2.0 * np.exp(-(t / td)**2) / td
        
        _T_photo = (L_bol / (4.0 * np.pi * sigmaSB * vej**2 * t**2))**0.25
        _R_photo = (L_bol / (4.0 * np.pi * sigmaSB * Tc**4))**0.5

        mask = _T_photo < Tc
        _T_photo[mask] = Tc
        mask = np.logical_not(mask)
        _R_photo[mask] = vej * t[mask]
        self.R_photo = _R_photo
        self.T_photo = _T_photo
    
    def evaluate(self, tvec_days, band):
        """
        Evaluate model at specific time values using the current parameters.

        Parameters
        ----------
        tvec_days : np.ndarray
            Time values
        band : string
            Band to evaluate
        """
        c = 2.99792458e10 # cm/s
        h = 6.626e-27 # erg * s
        kb = 1.38e-16 # erg/K
        Mpc = 3.08e24 # cm
        D = 1.0e-5 * Mpc # fiducial distance

        lmbda = self.lmbda_dict[band] * 1.0e-7 # convert to cm
        nu = c / lmbda

        B_nu = (2.0 * h * nu**3 / c**2) / np.expm1(h * nu / (kb * self.T_photo))
        F_nu = B_nu * np.pi * self.R_photo**2 / D**2
        
        mAB = -2.5 * np.log10(F_nu) - 48.6
        mask = np.isfinite(mAB)
        f = interp1d(self.tdays[mask], mAB[mask], fill_value="extrapolate")
        return f(tvec_days), self.sigma

