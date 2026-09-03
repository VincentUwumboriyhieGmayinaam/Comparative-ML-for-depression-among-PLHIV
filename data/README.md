# data/

This directory is intentionally empty in the published repository.

Individual participant data are not deposited here. See the "Data availability"
section of the top-level README for the reason and for the access procedure.

To reproduce the analysis locally, place the following files in this directory:

  - `Data CSV.csv`                 numeric-coded analysis dataset (264 x 78)
  - `Depression in HIV Data.csv`   label-coded companion file

then update the constant `D` at the top of `code/run.py` to point here.
