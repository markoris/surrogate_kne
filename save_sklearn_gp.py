#
# efficient_save_sklearn_gp.py
#
# GOAL
#   - sklearn gp objects I use always have a simple form
#   - this saves a json file holding *strings* describing the kernel, AND the X_train_ and y_train_ data used in the GP
#   - we *may* still  use a pickle for now, but it is *much* more efficient -- we're just saving the data we need
#   - this lets me change the data format to make it *portable*

import json

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy.linalg import cholesky, cho_solve

def gp2json(gp):
    """
    Converts gp object to (a) json object and (b) data for X_train_ and y_train
    """
    out_dict = {}
#    out_dict['kernel'] = str(gp.GP.gpr.kernel_)
    out_dict['kernel'] = [str(param) for param in gp.GP.gpr.kernel_.theta]
    out_dict['kernel_params'] = {}
    dict_params = gp.GP.gpr.kernel_.get_params()
    for name in dict_params:
        out_dict['kernel_params'][name] = str(dict_params[name])   # gives me ability to set more parameters in greater detail
    out_dict['y_train_std'] = str(gp.std)
    out_dict['y_train_mean'] = str(gp.mean)
    return [out_dict, gp.GP.gpr.X_train_, gp.GP.gpr.y_train_,gp.GP.gpr.alpha_]

def export_gp_compact(fname_base,gp):
    fname_json = fname_base+".json"
    fname_dat_X = fname_base+"_X.dat"
    fname_dat_y = fname_base+"_y.dat"
    fname_dat_alpha = fname_base+"_alpha.dat"
    my_json, my_X,my_y, my_alpha = gp2json(gp) 
    np.savetxt(fname_dat_X,my_X)
    np.savetxt(fname_dat_y,my_y)
    np.savetxt(fname_dat_alpha,my_alpha)
    with open(fname_json,'w') as f:
        json.dump(my_json, f)
    return None


def load_gp(fname_base):
    kernel=None
    with open(fname_base+".json",'r') as f:
        my_json = json.load(f)
    my_X = np.loadtxt(fname_base+"_X.dat")
    my_y = np.loadtxt(fname_base+"_y.dat")
    my_alpha = np.loadtxt(fname_base+"_alpha.dat")
    dict_params = my_json['kernel_params']
#    eval("kernel = "+my_json['kernel'])
#    kernel = eval(my_json['kernel'])
    theta = np.array(my_json['kernel']).astype('float')
    theta = np.power(np.e, theta)
    kernel = WhiteKernel(theta[0]) + theta[1]*RBF(length_scale=theta[2:])
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=0)
    gp.kernel_ = kernel
    dict_params_eval = {}
    for name in dict_params:
        if not('length' in name   or 'constant' in name):
            continue
        if name =="k2__k2__length_scale":
            one_space = ' '.join(dict_params[name].split())
            dict_params_eval[name] = eval(one_space.replace(' ',','))
        else:
            dict_params_eval[name] = eval(dict_params[name])
    gp.kernel_.set_params(**dict_params_eval)
    gp.X_train_ = my_X
    gp.y_train_ = my_y
    gp.alpha_ = my_alpha
    gp._y_train_std = float(my_json['y_train_std'])
    gp._y_train_mean = float(my_json['y_train_mean'])
    
    gp.predict = predict
    
    return gp

def predict(model, inputs, output=''):
    
    K = model.kernel_(model.X_train_) # setting up the kernel
    K[np.diag_indices_from(K)] += model.alpha
    model.L_ = cholesky(K, lower=True) # recalculating L matrix since this is what makes the pickled models bulky
    model._K_inv = None # has to be set to None so the GP knows to re-calculate matrices used for uncertainty
    K_trans = model.kernel_(inputs, model.X_train_)
    pred = K_trans.dot(model.alpha_)
    pred = model._y_train_std * pred + model._y_train_mean
    v = cho_solve((model.L_, True), K_trans.T)
    y_cov = model.kernel_(inputs) - K_trans.dot(v)
    err = np.sqrt(np.diag(y_cov))
    
    def lums_to_mags(data): # add err argument that handles conversion of errors
        '''
        assumes input of log10 luminosity
        '''
        r = 3.086e18 # parsec to cm
        r *= 10 # 10 pc for absolute magnitude
        flux = np.power(10, data) / (4 * np.pi * r**2)
        flux[np.where(flux <= 0)] = 1e-50 # eliminate zeroes with really small values
        mags = -48.6 - 2.5 * np.log10(flux)
        return mags
        
    if output == 'mags':
        return lums_to_mags(pred), lums_to_mags(err) # fix error propagation, currently incorrect
    
    if output == 'ABmags':
        wavs = np.array([476., 621., 754., 900., 1020., 1220., 1630., 2190.])
        ab_offsets = np.array([-0.103, 0.146, 0.366, 0.528, 0.634, 0.938, 1.379, 1.900]) # Hewett et al. 2006, MNRAS, 367, 454-468 Table 7
        offset_idx = np.argmin(np.abs([[inputs[i, -1]-wavs[j] for j in range(wavs.shape[0])] for i in range(inputs.shape[0])]), axis=1)
        # determines offset value by finding closest matching filter to user-specified wavelength
        return lums_to_mags(pred)+ab_offsets[offset_idx], lums_to_mags(err)
    
    return pred, err

                            
