# Radial profile grids

Place large HDF5 radial-profile grids here. The files are intentionally ignored
by Git.

Each model must provide `radius`, `rho` (or `logrho`), and `pressure` (or
`logP`). `temperature`, `mass_enclosed`, `mu`, `epsilon`, `grad_ad`, and
`grad_rad` improve the physical decomposition. Models may be stored as groups
of one-dimensional datasets or as root datasets shaped `[models, radius]`.
Global `mass`, `teff`, `metallicity`, and `age` values may be group attributes
or one-dimensional root datasets.

The public MIST EEP downloads are `.track.eep` evolutionary tables. They do not
contain radial density and pressure profiles and therefore cannot supervise
this PINN's radial outputs. Do not rename or convert those tables to HDF5 and
treat them as structure profiles.
