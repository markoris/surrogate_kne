# Surrogate kilonova model 

Another link to the surrogates and jupyter notebooks: https://github.com/markoris/surrogate_kne .


Edits for surrogate paper follow-up by Atul Kedia, Marko Ristic, Richard O'Shaughnessy et al. (on arXiv 2022):
Added separate Jupyter notebooks for a few tasks:
1) ``compare_training_data_tutorial_magnitude`` : Does everything that compare_training_data_tutorial does, but for magnitudes instead of luminosity.
2) ``compare_training_data_tutorial_wavelength`` : Does everything that compare_training_data_tutorial does, but also predicts the light curve interpolation for a new broadband filter in-between the initial sampling broadband filters. Specific parameters used in the example are:  (md, vd, mw, vw) = (0.014248, 0.183263, 0.085009, 0.052937) and wavelength of choice is = 1451 or 1839 nm. This also plots the direct SuperNu output light curves for the specific parameters through the files ‘luminosity_cfht.dat’ and ‘luminosity_f182m.dat’.

Several other minor updates were made to make the notebooks have fewer hardcoded parameters.


This repository provides a surrogate kilonova model developed by Marko Ristic, tuned to reproduce simulation models developed by Ryan Wollaeger and Oleg Korobkin.
Surrogates provide multiband output versus time and viewing angle, assuming axisymmetry. 
At present, the repository contains four surrogate models, associated with two companion papers, Kedia+ arXiv (2022) and Ristic+ Phys. Rev. Research (2022).

The tutorial files show how to load and use the data
* ``model_prediction_tutorial``: Demonstrates how to produce multiband output for an arbitrary point, including uncertainty bands.
* ``compare_training_data_tutorial``: Compares the underlying simulation data to our predictions
* ``time_angle_interpolation``: Illustrates our interpolation in time and angle, comparing to neighboring simulations



**Contact**:  If you have questions, contact Marko via this [Contact form](https://aspire.rit.edu/user/marko.ristic) or Atul Kedia at atulkedia93@gmail.com .



## Related software

**EM_PE**: [EM_PE](https://github.com/markoris/EM_PE) Parameter inference code for kilonova light curves used in our companion study.  This code already has modules which interface with our surrogate.

## Citation guide
If you use this work, please cite the companion paper and the hypercube study upon which this work is based:


```
@ARTICLE{more_surrogates,
  author={{Kedia}, Atul and {Ristic}, Marko and {O'Shaughnessy}, Richard and {Yelikar}, Anjali B. and {Wollaeger}, Ryan T. and {Korobkin}, Oleg and {Chase}, Eve A. and {Fryer}, Christopher L. and {Fontes}, Christopher J.},
  title ="{Surrogate light curve models for kilonovae with comprehensive wind ejecta outflows and parameter estimation for AT2017gfo}"
  journal={submitted},
}

@ARTICLE{surrogate_kne,
  author={{Ristic}, M. and {Champion}, E. and {O'Shaughnessy}, R. and {Wollaeger}, R. and {Korobkin}, O. and {Chase}, E.A. and {Fryer}, C.L. and {Hungerford}, A.L. and {Fontes}, C.J.},
  title ="{Interpolating detailed simulations of kilonovae: Adaptive learning and parameter inference applications}"
  journal={Physical Review Research},
  volume={4},
  number={1},
  pages={013046},
  year={2022},
  publisher={APS}
  url      = {https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.013046 }
}

@ARTICLE{rw_hypercube,
  author={{Wollaeger}, R. and {Fryer}, C.L. and Fontes, C.J. and Ristic, M. and Hungerford, A. and Korobkin, O. and {O'Shaughnessy}, R. and {Herring}, A.M.},
  title={A Broad Grid of 2D Kilonova Emission Models},
  journal={The Astrophysical Journal},
  volume={918},
  number={1},
  pages={10},
  year={2021},
  publisher={IOP Publishing},
  url={https://arxiv.org/abs/2105.11543 }
}

```
