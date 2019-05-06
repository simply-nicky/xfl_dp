# EXFEL data process repo (xfl_dp)

It's a package called xfl_dp written for post processing of XFEL AGIPD data preprocessed by cheetah. The package is written in Python 3.X.

## Features

The package performs triming out black images, intensity normalization, applying AGIPD geometry to stitch frames and writing frames and their corresponding pulseIDs and trainIDs in the cxi file.

## Required dependencies:

- NumPy
- h5py
- cfelpyutils

## How to use

You can import the package or use it as a command line tool:

```
$ python3 -m xfl_dp --help
usage: __main__.py [-h] [-n] [-v] rnum cnum tag [limit] [outpath]

Run XFEL post processing of cheetah data

positional arguments:
  rnum             run number
  cnum             cheetah number
  tag              cheetah tag associated with the current run (written after
                   a hyphen in the cheetah folder name)
  limit            minimum ADU value to trim out black images
  outpath          output folder location to write processed data

optional arguments:
  -h, --help       show this help message and exit
  -n, --normalize  normalize frame intensities
  -v, --verbosity  increase output verbosity

$ python -m xfl_dp -v 206 0 mll 500
List of typed arguments:
cnum=0
normalize=False
verbosity=True
tag=mll
limit=500
rnum=206
cheetah data is located in /gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r0206-mll/XFEL-r0206-c00.cxi
Writing data to folder: /Users/simply_nicky/OneDrive/programming/XFEL/hdf5/r0206-processed/XFEL-r0206-c01-processed.cxi
Done
```