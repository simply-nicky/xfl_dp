# EXFEL data process repo (xfl_dp)

It's a package called xfl_dp written for post processing of XFEL AGIPD data preprocessed by cheetah. The package is written in Python 3.X.

## Required dependencies:

- NumPy
- h5py

## How to use

You can import the package or use it as a command line tool:

```
$ python3 -m xfl_dp --help
usage: __main__.py [-h] [-v] rnum cnum tag limit

Run XFEL post processing of cheetah data

positional arguments:
  rnum             run number
  cnum             cheetah number
  tag              cheetah tag associated with the current run (written after
                   hyphen in cheetah folder name)
  limit            minimum ADU value to trim out black images

optional arguments:
  -h, --help       show this help message and exit
  -v, --verbosity  increase output verbosity
python -m xfl_dp -v 206 0 mll 500
```