# OIUTILS: read, display and model interferometric data in the OIFITS data format

## Overview

## model syntax

Possible keys in the model dictionary:

### Position:
  - 'x', 'y': define position in the field, in mas
      if not give, default will be 'x':0, 'y':0

### Size (will decide the model):
  - 'ud': uniform disk diameter (in mas)
  - or 'fwhm': Full width at half maximum for a Gaussian (in mas)
  - or 'udout': outside diameter for a ring (in mas) and 'thick': fractional thickness of the ring (0..1)

if none of these is given, the component will be fully resolved (V=0)

### Stretching:
  object can be stretched along one direction, using 2 additional parameters:
  - 'pa': projection angle from N (positive y) to E (positive x), in deg
  - 'i': inclination, in deg

### Slant:
  'slant', 'slant pa': definition TBD

### Rings:
  rings are by default uniform in brightness. This can be altered by using
  different keywords:
- 'profile': radial profile, can be 'uniform', 'doughnut' or 'power law'. If power law, 'power' gives the radial law
- 'ampi', 'phii': defines the cos variation amplitude and phase for i nodes along the azimuth

### Flux modeling:
if nothing is specified, flux is assume equal to 1 at all wavelengths

- 'f' or 'f0': if the constant flux (as function of wavelength). If not given, default will be 'f':1.0
- 'fi': polynomial amplitude in (wl-wl_min)**i (wl in um)
- 'fpow', 'famp': famp*(wl-wl_min)**fpow (wl in um)

spectral lines:
- 'line_i_f': amplitude of line i (>0 for emission, <0 for absorption)
- 'line_i_wl0': central wavelnegth of line i (um)
- 'line_i_gaussian': fwhm for Gaussian profile (warning: in nm, not um!)
   or 'line_i_lorentzian': width for Lorentzian (warning: in nm, not um!)
