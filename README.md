# OIUTILS: read, display and model interferometric data in the OIFITS data format

## Overview

This Python3 module allows to open, display and model astronomical interferometric
data stored in the OIFITS format. The most recent OIFITS format, based on the FITS
data format, is described in [Duvert et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract). The modeling is parametric, and allows least-square fitting, with bootstrapping estimation of uncertainties. For spectroscopic instruments (such as GRAVITY), some tools are provided to model and correct spectra for telluric lines.

The modeling of data is based on several principles:
- The model is composed of a combination of basic building blocks
- building blocks include uniform disks, uniform rings, gaussian, rings with arbitrary profiles and/or azimuthal variation.
- Building blocks can be deformed (analytically), including stretched in one preferred direction, or slanted.
- More complicated blocks are available, such as disks/rings with arbitrary radial profile, with possibility to include azimuthal variations.
- Each component has a spectrum, including modeling of emission or absorption lines (Gaussian or Lorentzian)
- In order for the computation to be fast (a requirement to perform data fitting), basic blocks have analytical complex visibilities. Moreover, for the same reason, their spectral component is independent of the geometry.

The principles are close to tolls such as [LITpro](https://www.jmmc.fr/english/tools/data-analysis/litpro). However, OIUTILS offers additional features:
- OIUTILS extends the modeling in the spectral dimension. For this reason, OIUTILS contains a module to do basic telluric correction (only for GRAVITY at the moment)
- Models' parameters can be expressed a function of others, which allows to build complex geometrical shapes.


## Install

Run the following command in the the root directory of the sources:
```
python setup.py install
```
if you do not have the root rights to your system, you can alternatively run:
```
python setup.py install --user
```

## Basic example

```
from oiutils import oifits, oimodels
data = oiutils.loadOI(files)
fit = oimodels.fitOI(data, param)
oimodels.showOI(data, fit)
```
In this example, user need to provide a list of OIFITS files in `files` and a first guess parameters dictionary to describe the model, for instance `param={'ud':1.0}` for a 1 milli-arcsecond angular uniform disk diameter.

## Advanced examples

The directory `examples` contains real life examples in the form of Jupyter notebooks:
- [Alpha Cen A](https://github.com/amerand/OIUTILS/blob/master/examples/alphaCenA.ipynb) PIONIER data from [Kervalla et al. A&A 597, 137 (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.137K/abstract). Fitting V2 with uniform disk or limb-darkened disks.
- [FU Ori](https://github.com/amerand/OIUTILS/blob/master/examples/FUOri.ipynb) GRAVITY data from [Liu et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884...97L/abstract). Fitting a 2 chromatic components model.


## Model syntax

Possible keys in the model dictionary:

### Position:
  - `x`, `y`: define position in the field, in mas
      if not give, default will be 'x':0, 'y':0

### Size (will decide the model):
  - `ud`: uniform disk diameter (in mas)
  - or `fwhm`: Full width at half maximum for a Gaussian (in mas)
  - or `diam`: outside diameter for a ring (in mas) and `thick`: fractional thickness of the ring (0..1)

if none of these is given, the component will be fully resolved (V=0)

### Stretching:
  Object can be stretched along one direction, using 2 additional parameters:
  - `pa`: projection angle from N (positive y) to E (positive x), in deg
  - `i`: inclination, in deg.

The dimensions will be reduced by a factor cos(i) along the direction perpendicular to the projection angle

### Slant:
  `slant`, `slant pa`: definition TBD

### Rings:
  Rings are by default uniform in brightness. This can be altered by using
  different keywords:
- `diam` is the diameter (in mas) and `thick` the thickness. By default, thickness is 1 if omitted.
- `profile`: radial profile, can be `uniform`, `doughnut`

Profiles can be arbitrary defined. To achieve this, you can define the profile as a string, as function of other parameters in the dictionary. There are also two special variables: `R` and `MU` where `R`=2*r/`diam` and `MU`=sqrt(1-`R`^2). For example, to model a limb-darkened disk with a power law profile (in `MU`), one will define the model as:
```
param = {'diam':1.0, 'profile':'MU**alpha', 'alpha':0.1}
```
The parsing of `profile` is very basic, so do not create variable names with common name. For instance, using `{'profile':'1-s*np.sqrt(MU)', 's':0.1}` will fail, since the profiles will be transformed into `1-0.1*np.0.1qrt(MU)` (the 's' of sqrt will be substituted with its value of 0.1).

- `ampi`, `phii`: defines the cos variation amplitude and phase for i nodes along the azimuth

TBD

### Flux modeling:
If nothing is specified, flux is assume equal to 1 at all wavelengths. Flux can be described using several parameters:

- `f0`: if the constant flux (as function of wavelength). If not given, default will be 'f':1.0
- `fi`: polynomial amplitude in (wl-wl_min)^i (wl in um)

More complex chromatic variations can be achieved using the `spectrum` parameter, very much like the `profile` parameter for rings. Note that the spectrum will be added to any over spectral information present in the parameters. The following are equivalent:
```
params1 = {'f0':1.0, 'f2':0.2}
params2 = {'f0':1.0, 'A2':0.2, 'spectrum':'A2*(WL-np.min(WL))**2'}
params3 = {'A0':1.0, 'A2':0.2, 'spectrum':'A0 + A2*(WL-np.min(WL))**2'}
```
You should not use parameter names (the `A0`, `A2` in the example above) which conflict with expected parameters such as `f0`, `f2`. etc.

Spectral lines:
- `line_i_f`: amplitude of line i (>0 for emission, <0 for absorption)
- `line_i_wl0`: central wavelnegth of line i (um)
- `line_i_gaussian`: fwhm for Gaussian profile (warning: in nm, not um!)
   or `line_i_lorentzian`: width for Lorentzian (warning: in nm, not um!)
