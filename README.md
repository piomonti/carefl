# Code for "Causal Autoregressive Flow"


This repository contains code to reproduce results presented in "Autoregressive flow-based causal
discovery and inference", presented at the 2nd ICML workshop on Invertible Neural Networks,
Normalizing Flows, and Explicit Likelihood Models (2020).

The `main.py` script is the main gateway to reproduce the experiments detailed in the mansucript, and is straightforward
to use. Type `python main.py -h` to learn about the options.

Hyperparameters can be changed through the configuration files under `configs/`.
The `main.py` is setup to read the corresponding config file for each experiment, but this can be overwritten suing the 
`-y` or `--config` flag.

### Dependencies
This project was tested with the following versions:

- python 3.7
- numpy 1.18.2
- pytorch 1.4
- scikit-learn 0.22.2
- scipy 1.4.1
- matplotlib 3.2.1
- seaborn 0.10

This project uses normalizing flows implementation from [this](https://github.com/karpathy/pytorch-normalizing-flows) library.

### Slurm usage
Experiments where run using the SLURM system. The `slurm_main_cpu.sbatch` is used to run jobs on CPU, and
 `slurm_main.sbatch` for the GPU.

To run simulations in parallel:
```bash
for SIZE in 25 50 75 100 150 250 500; do
    for ALGO in lrhyv reci anm; do
        for DSET in linear hoyer2009 nueralnet_l1 mnm veryhighdim; do
            sbatch slurm_main_cpu.sbatch -s -m $DSET -a $ALGO -n $SIZE
        done
    done
done
ALGO=carefl
for SIZE in 25 50 75 100 150 250 500; do
    for DSET in linear hoyer2009 nueralnet_l1 mnm veryhighdim; do
        sbatch slurm_main_cpu.sbatch -s -m $DSET -a $ALGO -n $SIZE
    done
done

```

___
To run interventions:
```bash
for SIZE in 250 500 750 1000 1250 1500 2000 2500; do
    for ALGO in gp linear; do
        sbatch slurm_main_cpu.sbatch -i -a $ALGO -n $SIZE
    done
done
ALGO=carefl
for SIZE in 250 500 750 1000 1250 1500 2000 2500; do
    sbatch slurm_main_cpu.sbatch -i -a $ALGO -n $SIZE
done
```

___
To run arrow of time on EEG data:
```bash
for ALGO in LRHyv RECI ANM; do
    for IDX in {0..117}; do
        sbatch slurm_main_cpu.sbatch -e -n $IDX -a $ALGO --n-sims 11
    done
done
ALGO=carefl
for IDX in {0..117}; do
    sbatch slurm_main.sbatch -e -n $IDX -a $ALGO --n-sims 11
done
```

___
To run interventions on fMRI data (this experiment outputs to standard output):
```bash
python main.py -f
```

___
To run pairs:
```bash
for IDX in {1..108}; do
    sbatch slurm_main_cpu.sbatch -p -n $IDX --n-sims 10
done
```


