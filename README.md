# sr-downscaling-demo
Lightweight demo of a recurrent super-resolution CNN for accelerating subsurface reactive transport simulations. Developed with The University of Adelaide, IFPEN and CSIRO. Includes example dataset, network code and training scripts for reproducing coarse-to-fine downscaling.
This repository contains a lightweight, fully reproducible demonstration of the workflow developed during collaborative research between The University of Adelaide and IFP Energies Nouvelles (IFPEN). The work was initiated during a sabbatical hosted at IFPEN and focuses on accelerating subsurface flow simulations through deep learning.

Forecasting the fate of injected fluids in geological formations (e.g., CO₂ storage) often requires computationally expensive fine-scale multi-physics simulations, especially when accounting for geological uncertainty. To address this challenge, we developed a recurrent super-resolution convolutional neural network that maps coarse-scale simulation outputs to fine-scale predictions, recovering information lost through upscaling.

The method:
Uses permeability (or other geological parameters), the fine-scale state at the current timestep and the upsampled coarse-scale state at the next timestep as inputs. 
Treats all variables as multi-channel images and embeds temporal recurrence so that each timestep of each realisation becomes a training sample.
Can be integrated into uncertainty quantification (UQ) or history matching workflows to deliver large speed-ups while retaining fine-scale resolution.

This repository provides a small synthetic example, the network definition, and training code to reproduce the workflow on a CPU.

If you use this repository, please cite the associated publication:

Sayyafzadeh, M., Bouquet, S., & Gervais, V. (2024, September).  
**Downscaling State Variables of Reactive Transport Simulation Using Recurrent Super-Resolution Networks.**  
In *ECMOR 2024* (Vol. 2024, No. 1, pp. 1–18).  
European Association of Geoscientists & Engineers.  
https://doi.org/10.3997/2214-4609.202452072


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


Dataset description
Type: Two-phase, two-component immiscible flow.
Permeability fields: Random MPS realisations based on the Strebelle (2002) training image, with no vertical continuity between layers.
Initial condition: Fully saturated with the non-wetting phase.
Boundary conditions: Wetting phase injected at a constant rate of 100 m³/day per unit area on one side; opposite side free boundary; remaining sides no-flow.
Fine-scale model: 159×159×2 cells, isotropic permeability.
Coarse-scale model: 53×53×1 cells. 

Upscaling: Local pressure solver with periodic boundary conditions (MRST Upscaling module).

Simulation: Fully implicit scheme in OPM Flow. 

Primary variables: Wetting phase saturation and non-wetting phase pressure (no secondary variables).

Input parameters: Fine-scale permeability fields (other parameters fixed across all realisations).

Total realisations/samples: 100
Training set: 80 realisations
Test set: 20 realisations
Training/validation samples: 720 (derived from timesteps #2 onward to avoid large disturbances in the initial timestep that could degrade network performance).

Performance note
The coarse-scale simulation runs way faster compared to the fine-scale model. The reduction from two layers to one in the coarse grid introduces a non-trivial challenge for restoring lost vertical information, making this example a suitable testbed for the proposed recurrent super-resolution network.



Create the env:
conda env create -f env_macos.yml   
