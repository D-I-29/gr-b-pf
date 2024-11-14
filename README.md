# gr-b-pf
# Gutenberg-Richter law $b$ value estimation using Particle Filter

## About this
This repository provides code for estimation of Gutenberg-Richter law $b$ value using Particle Filter. 

## Tutorial
Example of estimating $b$ value is displayed by [notebook](https://github.com/D-I-29/gr-b-pf/blob/main/notebook/notebook_demo.ipynb).

## Quick Start
You can set up calculating environment and run code as following.
1. setting up environment
Install python(3.10.7+) packages by
```
pip install -r requirements.txt
```

If you use Docker environment, you can start using `Dockerfile` by
```
docker build -t pfgr .
docker run --name pfgr-c -it pfgr /bin/bash
```

2. run code
If you have data (CSV format) with column `date_time`, `magnitude`, you can run code as following
```
python3 run.py [--data DATA] [--num_particle NUM_PARTICLE] [--m_lower M_LOWER]
```

3. results
Results of estimating $b$ value are put at the directory `./result/` as CSV file.

## Models
Models implemented in this code are described at our [paper](https://www.nature.com/articles/s41598-024-54576-x). Below models are supported.

##### Type of GR law
- Exponential distribution
- Truncated GR distribution
##### Type of variation of $b$ value
- Random walk model
- Random walk with truncated Normal distribution model
