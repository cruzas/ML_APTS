# ML_APTS
Additively preconditioned trust-region strategies (APTS) for machine learning (ML)

## Authors
* Samuel A. Cruz Alegría (Euler Institute at Università della Svizzera italiana; UniDistance Suisse).
* Ken Trotti (Euler Insitute at Università della Svizzera italiana).
* Alena Kopaničáková (Brown University; Euler Institute at Università della Svizzera italiana).
* Prof. Dr. Rolf Krause (Euler Institute at Università della Svizzera italiana; UniDistance Suisse).

In collaboration with Daniel Ganellari and Sebastian Keller at the Swiss Nattional Supercomputing Centre (CSCS).

## Funding
This work was supported by the Swiss Platform for Advanced Scientific Computing (PASC) project **ExaTrain** (funding periods 2017-2021 and 2021-2024) and by the Swiss National Science Foundation through the projects "ML<sup>2</sup> -- Multilevel and Domain Decomposition Methods for Machine Learning" (197041) and "Multilevel training of DeepONets -- multiscale and multiphysics applications" (206745). 

## Usage
All code resides in the `src` folder. The code can be executed as `python main.py`. All of the default values are configured inside `./src/utils/utility.py`, in the function `parse_args()`.

The optimizer `APTS_W` can be executed on a single computer, as its implementation is currently sequential. The optimizer `APTS_D` was designed to be executed on a cluster of computers in parallel, hence, running our code with `APTS_D` is currently not possible on a single computer with one or more GPUs.
