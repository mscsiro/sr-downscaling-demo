# sr-downscaling-demo
Lightweight demo of a recurrent super-resolution CNN for accelerating subsurface reactive transport simulations. Developed with The University of Adelaide, IFPEN and CSIRO. Includes example dataset, network code and training scripts for reproducing coarse-to-fine downscaling.
This repository contains a lightweight, fully reproducible demonstration of the workflow developed during collaborative research between The University of Adelaide and IFP Energies Nouvelles (IFPEN). The work was initiated during a sabbatical hosted at IFPEN and focuses on accelerating subsurface flow simulations through deep learning.

Forecasting the fate of injected fluids in geological formations (e.g., CO₂ storage) often requires computationally expensive fine-scale multi-physics simulations, especially when accounting for geological uncertainty. To address this challenge, we developed a recurrent super-resolution convolutional neural network that maps coarse-scale simulation outputs to fine-scale predictions, recovering information lost through upscaling.

The method:
Uses permeability (or other geological parameters), the fine-scale state at the current timestep and the upsampled coarse-scale state at the next timestep as inputs. 
Treats all variables as multi-channel images and embeds temporal recurrence so that each timestep of each realisation becomes a training sample.
Can be integrated into uncertainty quantification (UQ) or history matching workflows to deliver large speed-ups while retaining fine-scale resolution.

This repository provides a small synthetic example, the network definition, and training code to reproduce the workflow on a CPU.

Data provenance & citations
The demonstration dataset included in this repository is derived from publicly available tools and methods:

Multiple-Point Geostatistics training image
Strebelle, S. (2002). Conditional simulation of complex geological structures using multiple-point statistics. Mathematical Geology, 34(1), 1–21.
https://doi.org/10.1023/A:1014009426274

Upscaling performed using the MATLAB Reservoir Simulation Toolbox (MRST):
Lie, K.‐A. (2019). An Introduction to Reservoir Simulation Using MATLAB/GNU Octave: User Guide for the MATLAB Reservoir Simulation Toolbox (MRST). Cambridge University Press.
https://www.sintef.no/projectweb/mrst/

Flow simulation performed with OPM Flow:
Open Porous Media (OPM) Initiative. OPM Flow – A fully-implicit black-oil simulator.
https://opm-project.org
