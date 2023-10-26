# actrys (Automated Calibration for Traffic Simulations)

[![Build Status](https://github.com/vishalmhjn/actrys/actions/workflows/main.yml/badge.svg?branch=main&event=push)](https://github.com/vishalmhjn/actrys/actions/workflows/main.yml)

**actrys** is a Python-based platform designed for calibrating traffic simulations in SUMO (Simulation of Urban Mobility).

<p align="center">
  <img src="resources/munich_traffic_flows.gif" alt="Traffic Visualization" width="450"/>
</p>

<p align="center">
  <img src="resources/munich_network_speeds.gif" alt="Traffic Visualization" width="500"/>
</p>

## Overview

This repository includes code for:

- **Input/Output Interfaces**: To interact with SUMO, initialize and simulate scenarios, and collect output data.
- **Calibration Components**:
  - Optimization Strategies: SPSA, W-SPSA, Bayesian optimization.
  - Goodness-of-Fit Criteria.
- **Assignment Matrix Extraction**: Based on the simulated routes.
- **Synthetic Simulator**: For prototyping with static traffic assignment.
- **Utilities**: For preparing and processing SUMO input files, such as network downloading, trip filtering, adding detectors, and file format conversion.
- **Plotting Tools**.

## Framework

The platform follows a step-wise approach for sequential calibration of:

- Demand parameters or Origin-Destination (OD) flows.
- Supply parameters for mesoscopic simulations.

To achieve this, the following process is followed:

1. **Bias-Correction in OD Matrices**: Using a one-shot heuristic.
2. **Bayesian Optimization**: Fine-tuning SPSA parameters using an analytical or static assignment matrix approximated from the simulator.
3. **W-SPSA with Ensembling Techniques**: Including cold and warm restarts.
4. **Supply Calibration** using Bayesian optimization.

Currently, the platform can handle link-based Measures of Performance (MOP) such as link traffic counts and link speeds in the calibration process.

This platform can calibrate two types of scenarios:

- **Analytical or Static Simulator**: Used for prototyping, this scenario generates a random assignment matrix controlled by different parameters to map OD flows to link counts. No external data files are needed to run this scenario, except for a few parameters that control the scenario properties.

- **Black-box or Dynamic Simulator (SUMO)**: This scenario includes:
  - Creation of trips from time-dependent OD matrices.
  - Routing for trips based on the route-choice algorithm.
  - Dynamic network conditions, traffic propagation, and re-routing.

## Architecture

<p align="center">
<img src="resources/architecture.png" alt="architecture" width="500" align="center"/>
</p>

## Input Preparation for SUMO Simulations

**Black-box or dynamic simulator** scenarios require several formatted inputs. Example files are provided in the [munich](munich/) directory:

- **Network File**: A standard SUMO network [(Example)](munich/network.net.xml).
- **OD Matrices**: Time-dependent specification of trips between Origin-Destination zones [(Example)](munich/demand/).
- **Link Sensors**: A file specifying the properties (location and data collection frequency) of edge or link sensors [(Example)](munich/additional.add.xml).
- **Traffic Analysis Zones**: Mapping between origin-destination zones and network edges [(Example)](munich/tazes.taz.xml).

## Custom Synthetic Scenario with SUMO

For a custom synthetic scenario, the following process is followed:

1. A true demand is simulated to obtain "_real sensor measurements_."
2. Subsequently, the true demand is perturbed by the addition of bias and variance.
3. Perturbed demand is simulated, and corresponding simulation sensor measurements are compared with the "_real sensor measurements_." Based on the discrepancy between real and simulated data, the calibrator aims to recover the true demand matrix.

## Custom Real Scenario with SUMO

For a real scenario, you can use real sensor data such as link volumes or link speeds. These measurements could be obtained from open data sources, traffic operator websites, or city open data portals. In this case, additional input is needed to use the observed counts as an input:

1. **Observed Sensor Data**: Such as link volumes or link speeds to be used as Measures of Performance (MOP) [(Example)](munich/realdata.csv).

2. In a real scenario, the true demand is not known. What you have is an initial demand matrix, which is simulated. Corresponding simulation sensor measurements are compared with the real sensor measurements. Based on the discrepancy between real and simulated data, the calibrator aims to recover the true demand matrix.

## Requirements

The framework has been tested on **SUMO 1.13.0** and **Python 3.8** on both Ubuntu 18.04 and macOS 13.2. To set up your environment, follow these steps:

1. Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv) using the following commands:

   ```sh
   cd
   python3 -m venv actrys
   ```

2. Then activate the virtual environment using

   ```sh
   source ~/actrys/bin/activate
   ```

3. Make the parent folder of the repository your current directory:

   ```sh
   cd path/to/your/github/repository/actrys
   ```

4. Install the Python [requirements](requirements.txt) using:
   ```sh
   pip install -r requirements.txt
   ```

## Execution

### Running Analytical or Static Simulation Scenarios

1. Set the current working directory to [src/wrapper](src/wrapper/) in the Command Line Interface (CLI):

   ```sh
   cd src/wrapper
   ```

2. The Python file serving as the overall wrapper is [run_analytical_sim.py](src/wrapper/run_analytical_sim.py), which calls the [synthetic_calibrator.py](src/core/synthetic_calibrator.py).

3. The scenario parameters such as number of OD pairs and detectors can be changed in the [synthetic_calibrator.py](src/core/synthetic_calibrator.py) directly.

4. Run the following command in the terminal:

   ```sh
   python runAnalyticalSim.py
   ```

5. Outputs are stored in [synthetic_sims](synthetic_sims/) folder

### Running Black-box or Dynamic Simulation Scenarios with Synthetic Counts

1. First, specify the paths in [paths](src/core/paths.py) and [parameters](src/core/params.py). Please note that the platform has not been tested on Windows OS, so paths may need to be adapted.

2. Set the path to the SUMO folder with the SUMO_HOME variable. For example, on macOS, it is generally stored at the following path for SUMO version 1.10.0. More details can be found in this [link](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#sumo_home) for more details.

   ```sh
   export SUMO_HOME="/usr/local/Cellar/sumo/1.10.0/share/sumo"
   ```

3. Change the current working directory to [src/wrapper](src/wrapper/) in the command line interface:

   ```sh
   cd src/wrapper
   ```

4. The Python file serving as the overall wrapper is [run_sim.py](src/wrapper/run_sim.py), which calls the secondary wrapper [wrapper.sh](src/wrapper/wrapper.sh). Before running the simulation for synthetic counts, set the variable **synthetic_counts** as True in [run_sim.py](src/wrapper/run_sim.py).

5. Run the following command in the terminal to run the simulation (e.g., for Munich):

   ```sh
   python runSim.py munich
   ```

6. Outputs are stored in [munich](munich/) folder

### Following procedure for running <b>Black-box or dynamic simulation </b> scenarios with real-world counts:

1. Specify the paths to real data in [wrapper.sh](src/wrapper/wrapper.sh). For example:

   ```sh
   export FILE_MATCH_DETECTORS=../../$SCENARIO/ sample_real_world_data/matchable_detectors.csv
   export FILE_REAL_COUNTS=../../$SCENARIO/sample_real_world_data/dummy_counts_data.csv
   export FILE_REAL_SPEEDS=../../$SCENARIO/sample_real_world_data/dummy_speed_data.csv
   ```

2. Specify the detector identifier in the [params](src/core/params.py) as per the [detector file](munich/additional.add.xml) (_<e1Detector id=""_).

   ```
   additonal_identifier = "e1Detector_id"
   output_identifier = "interval_id"
   ```

3. Before running the simulation scenario, set the variable **synthetic_counts** as False in [run_sim.py](src/wrapper/run_sim.py). This is because, we want to use the real data in this case.

4. Finally, run the following command in the terminal to run the simulation for the **Munich** scenario:

   ```sh
   python runSim.py munich
   ```

## Citation

If you use these codes in your work, kindly cite the following preprint:

Mahajan, V., Cantelmo, G., and Antoniou, C, Towards automated calibration of large-scale traffic simulations, [preprint_v2](https://mediatum.ub.tum.de/doc/1701188/1701188.pdf), 2023.

## Acknowledgements

1. SUMO: https://github.com/eclipse/sumo
2. Noisyopt library: https://github.com/andim/noisyopt
