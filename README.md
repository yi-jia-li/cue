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
from cue.continuum import predict as cont_predict
par = [[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]]
lines = line_predict(theta=par).nn_predict()
cont = cont_predict(theta=par).nn_predict()
```

An example of fitting emission lines with cue in [demo/cue_demo.ipynb](https://github.com/yi-jia-li/cue/blob/main/demo/cue_demo.ipynb)


A list of the emulated lines can be found in [data/lineList.dat](https://github.com/yi-jia-li/cue/blob/main/src/cue/data/lineList.dat)

## Using `cuejax`

To install:

```
cd <install_dir>
git clone https://github.com/efburnham/cue.git
cd cue
pip install .
```

To use in python:

```
import cuejax as cue
emul = cue.Emulator()
lines = emul.predict_lines(theta)
cont = emul.predict_cont(theta)
```

## Citation

If you use this code, please reference [this paper](https://ui.adsabs.harvard.edu/abs/2024arXiv240504598L/abstract):
```
@ARTICLE{2024arXiv240504598L,
       author = {{Li}, Yijia and {Leja}, Joel and {Johnson}, Benjamin D. and {Tacchella}, Sandro and {Davies}, Rebecca and {Belli}, Sirio and {Park}, Minjung and {Emami}, Razieh},
        title = "{Cue: A Fast and Flexible Photoionization Emulator for Modeling Nebular Emission Powered By Almost Any Ionizing Source}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = may,
          eid = {arXiv:2405.04598},
        pages = {arXiv:2405.04598},
          doi = {10.48550/arXiv.2405.04598},
archivePrefix = {arXiv},
       eprint = {2405.04598},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240504598L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
