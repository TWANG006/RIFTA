## Introduction
RIFTA is the Robust Iterative Fourier Transform based dwell time Algorithm (RIFTA) for deterministic optics fabrication. RIFTA was proposed and developed at the National Synchrotron Light Source II (NSLS-II), NY, USA, for the Ion Beam Figuring (IBF) of synchrotron X-ray mirrors. It can be generally applied to any Computer Controlled Optical Surfacing (CCOS) processes. Both the MATLAB and Python implementations of the algorithm are open sourced in this repository.

## Implemented algorithms
- Surface height-based RIFTA [1]
- Surface slope-based RIFTA
- Thresholded inverse filtering assisted by direct search [1]
- 2D convolution using Fast Fourier Transform (FFT)

## RIFTA example results
![RIFTA results](/image/RIFTA_results.png)

## Note
- All the units used in the code are metres unless otherwise specified.
- The common arguments that are required for IBFest dwell time calculation include:
  - Tool Influence Function (TIF): the TIF can come from either the measurement or model, by choosing ```avg``` or ```model```, respectively. If ```model``` is chosen, the parameters for a 2D Gaussian should be set, includeing the Peak Removal Rate (PRR) ```A```, the ```Sigma```, the diameter ```d```, and the centers ```u```. If ```avg``` is chosen, ```Xtif```, ```Ytif```, and ```Ztif``` should be provided. 
  - ```Zd```: the desired height to be removed. 
  - ```ca_range```: the range of the Clear Aperture (CA), which is a struct contains ```x_s```, ```y_s```, ```x_e```, and ```y_e```, which are the start and end coordinates (units are meters) of the CA. 
  - ```dg_range```: the range of the Dwell Grid (DG), which should be larger than ```ca_range``` at least the radius of the BRF on each side.

## Reference
[1] [Wang, T., Huang, L., Kang, H. et al. RIFTA: A Robust Iterative Fourier Transform-based dwell time Algorithm for ultra-precision ion beam figuring of synchrotron mirrors. Sci Rep 10, 8135 (2020). https://doi.org/10.1038/s41598-020-64923-3.](https://doi.org/10.1038/s41598-020-64923-3)
