import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kn_interp_angle import kn_interp_angle

model = kn_interp_angle()

params = {"mej_dyn":0.001, "vej_dyn":0.3, "mej_wind":0.01, "vej_wind":0.05, "theta":30.0}

model.set_params(params, (0, 0))

t = np.logspace(np.log10(0.1), np.log10(7.5), 20)
b = "g"
mag, err = model.evaluate(t, b)

plt.plot(t, mag)
plt.xscale("log")
plt.gca().invert_yaxis()

plt.savefig("kn_interp_test.png")
