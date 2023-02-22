# actrys (Automated Calibration for TRaffic SImulations)
Copyright 2020-2023 Vishal Mahajan

<b>actrys</b> is a Python-based platform to calibrate the traffic simulations in SUMO. 

<p align="center">
<img src="resources/munich_traffic_flows.gif" alt="drawing" width="400" align="center"/>
</p>

## Overview

This repository contains code for:
* Input/ Ouptut interfaces to call SUMO, initialize ad simulate the scenarios, and collect the output data.
* Calibration components:
    * Optimization strategy: SPSA, W-SPSA, Bayesian optimization
    * Goodness-of-fit criterias
* Assignment matrix extraction based on the simulated routes
* Synthetic simulator with a static traffic assignment for prototyping.
* Utilities
    * Preparing and processing SUMO input files such as network downloading, trip filtering, adding detectors, file format conversion
    * Plotting

<!-- ## Analytical or static simulator


## SUMO simulator -->

## Citation
If you use these codes in your work, kindly cite the following working paper:

Mahajan, V., Cantelmo, G., and Antoniou, C, One-shot heuristic and ensembling for automated calibration of large-scale traffic simulations, Working Paper, 2023.



## Acknowledgements
1. SUMO: https://github.com/eclipse/sumo
2. Noisyopt library: https://github.com/andim/noisyopt
