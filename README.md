# Surrogate kilonova model 

This repository provides a surrogate kilonova model developed by M. Ristic, tuned to reproduce simulation models developed by Ryan Wollaeger and Oleg Korobkin.
Surrogates provide multiband output versus time and viewing angle, assuming axisymmetry. 
At present, the repository contains one surrogate model, associated with a companion paper.

The tutorial files show how to load and use the data
* ``model_prediction_tutorial``: Demonstrates how to produce multiband output for an arbitrary point, including uncertainty bands.
* ``compare_training_data_tutorial``: Compares the underlying simulation data to our predictions
* ``time_angle_interpolation``: Illustrates our interpolation in time and angle, comparing to neighboring simulations



**Contact**:  If you have questions, contact Marko via this [Contact form](https://aspire.rit.edu/user/marko.ristic)



## Related software

**EM_PE**: [EM_PE](https://github.com/bwc3252/EM_PE) Parameter inference code for kilonova light curves used in our companion study.  This code already has modules which interface with our surrogate.

## Citation guide
If you use this work, please cite the companion paper and the hypercube study upon which this work is based:


```
@ARTICLE{surrogate_kne,
  author={{Ristic}, M. and {Champion}, E. and {O'Shaughnessy}, R. and {Wollaeger}, R. and {Korobkin}, O. and {Chase}, E.A. and {Fryer}, C.L. and {Hungerford}, A.L. and {Fontes}, C.J.},
  title ="{Interpolating detailed simulations of kilonovae: adaptive learning and parameter inference applications}"
  year=2021,
  journal = {Submitted},
   url      = {http://xxx.lanl.gov/abs/arXiv:2105.07013}
}

@ARTICLE{rw_hypercube,
  author={{Wollaeger}, R. and {Fryer}, C.L. and Fontes, C.J. and Ristic, M. and Hungerford, A. and Korobkin, O. and {O'Shaughnessy}, R. and {Herring}, A.M.},
  title={A Broad Grid of 2D Kilonova Emission Models},
  year=2021,
  journal = {Submitted to \apj}
}

```
