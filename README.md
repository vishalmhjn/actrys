# actrys (Automated Calibration for TRaffic SImulations)
Copyright 2020-2023 Vishal Mahajan

<b>actrys</b> is a Python-based platform to calibrate the traffic simulations in SUMO. 

<p align="center">
<img src="resources/munich_traffic_flows.gif" alt="traffic_visualization" width="400" align="center"/>
</p>

## Overview
This repository contains code for:
* Input/ Ouptut interfaces to call SUMO, initialize and simulate the scenarios, and collect the output data.
* Calibration components:
    * Optimization strategy: SPSA, W-SPSA, Bayesian optimization
    * Goodness-of-fit criterias
* Assignment matrix extraction based on the simulated routes
* Synthetic simulator with a static traffic assignment for prototyping.
* Utilities
    * Preparing and processing SUMO input files such as network downloading, trip filtering, adding detectors, file format conversion
    * Plotting

## Framework
The platform implements a step-wise approach for sequential calibration of
* Demand parameters or Origin-Destination (OD) flows
* Supply parameters for a mesoscopic simulation

To achieve this, following process is followed:
* Bias-correction in OD matrix using one-shot heuristic
* Bayesian optimization to fine-tune SPSA parameters using analytical or static assignment matrix approximated from the simulator
* W-SPSA with ensembling techniques with cold and warm restarts
* Supply calibration using Bayesian optimization

Currently, the platform can handle link based Measures of Performance (MOP) such as link traffic counts and link speeds in the calibration.

This platform can calibrate two kinds of scenarios:
* <b>Analytical or static simulator: </b> This is used for prototyping. Here, the assignment matrix is a randomly generated matrix. This random assignment matrix is controlled by different parameters and maps the OD flows to link counts. No external file or data are needed to run this scenario, except few parameters which control the scenario properties.
* <b>Black-box or dynamic simulator: </b> SUMO is used in this scenario. The simulation encompasses:
    * Creation of trips from time-dependent OD matrices
    * Routing for the trips based on the route-choice algorithm
    * Dynamic network conditions, traffic propagation and re-routing

## Architecture

<p align="center">
<img src="resources/architecture.png" alt="architecture" width="500" align="center"/>
</p>

## Input preparation
<b>Black-box or dynamic simulator </b> scenarios need quite-a-few formatted inputs
* Network file: Standard SUMO network
* OD matrices inputs: Time-dependent specification of trips between Origin-Destination zones
* Link sensor file inputs: File specifying the properties (location, frequency of data collection) of edge or link sensors
* Traffic Analysis Zone file: Mapping between origin-destination zones and network edges
* Observed sensor data such as link volumes or link speeds to be used as MOP


## Custom Synthetic Scenario
_todo_

## Custom Real Scenario
_todo_

## Requirements
* The framework has been tested on __SUMO 1.13.0__ and __Python 3.8__ on both Ubuntu 18.04 and macOS 13.2. You should create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) using
	```sh
	cd 
	python3 -m venv actrys
	```
* Then activate the virtual environment using
	```sh
	source ~/actrys/bin/activate
	```
* Make the github repository your current directory:
	```sh
	cd path/to/your/github/repository
	```
* Install the Python [requirements](requirements.txt) using:
	```sh
	pip install -r requirements.txt
	```
## Execution
Following procedure for running <b>Analytical or static simulation </b> scenarios. 

* Set the current working director to [src/wrapper](src/wrapper/) in the Command line interface, as above.

* The Python file is the overall wrapper [run_analytical_sim.py](src/wrapper/run_analytical_sim.py), which calls the [synthetic_calibrator.py](src/core/synthetic_calibrator.py).

* Run the following in terminal:
	```sh
	python run_analytical_sim.py
	```
* Outputs are stored in [synthetic_sims](synthetic_sims/) folder

Following procedure for running <b>Black-box or dynamic simulation </b> scenarios with synthetic counts:

* First, [paths](src/core/paths.py) and [parameters](src/core/params.py) are to be specified. The platform has not been tested on Windows OS, so the Paths might need to be adapted.

* Set path to the SUMO folder with SUMO_HOME variable. for example in OSX, it is generally stored at following path for SUMO version 1.10.0. You can check the this [link](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) for more details.

	```sh
	export SUMO_HOME="/usr/local/Cellar/sumo/1.10.0/share/sumo"
	```

* Then, the current working director should be changed to [src/wrapper](src/wrapper/) in the Command line interface. You can set this from home directory by 
	```sh
	cd src/wrapper
	```

* The Python file is the overall wrapper [run_sim.py](src/wrapper/run_sim.py), which calls the secondary wrapper [wrapper.sh](src/wrapper/wrapper.sh). Run the following in terminal:
	```sh
	python run_sim.py munich
	```
* Outputs are stored in [munich](munich/) folder

Following procedure for running <b>Black-box or dynamic simulation </b> scenarios with real-world counts:
## Citation
If you use these codes in your work, kindly cite the following preprint:

Mahajan, V., Cantelmo, G., and Antoniou, C, One-shot heuristic and ensembling for automated calibration of large-scale traffic simulations, [preprint](https://mediatum.ub.tum.de/doc/1701188/1701188.pdf), 2023.

## Acknowledgements
1. SUMO: https://github.com/eclipse/sumo
2. Noisyopt library: https://github.com/andim/noisyopt

## Disclaimer
The codebase is being improved for readability and execution. Please excuse the prototypical nature of the platform.
