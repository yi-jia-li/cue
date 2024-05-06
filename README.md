# Cue
Cue is a flexible tool for interpreting nebular emission across a wide range of ionizing conditions of galaxies. It is a neural net emulator based on the Cloudy photoionization code. Cue does not require a specific ionizing spectrum as a source, instead approximating the ionizing spectrum with a 4-part piece-wise power-law. Along with the flexible ionizing spectra, Cue allows freedom in [O/H], [N/O], [C/O], gas density, and total ionizing photon budget.

## Install
git clone https://github.com/yi-jia-li/cue.git

cd cue

python -m pip install .

## Uninstall

pip uninstall astro-cue

## How to use the code

An example of making the nebular line and continuum prediction based on the cue parameters: 
```
from cue.line import predict as line_predict
from cue.cont import predict as cont_predict
par = [[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]]
lines = line_predict(theta=par).nn_predict()
cont = cont_predict(theta=par).nn_predict()
```

An example of fitting emission lines with cue in [demo/cue_demo.ipynb](https://github.com/yi-jia-li/cue/blob/main/demo/cue_demo.ipynb)


A list of the emulated lines can be found in [data/lineList.dat](https://github.com/yi-jia-li/cue/blob/main/src/cue/data/lineList.dat)
