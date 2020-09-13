# Code for "Autoregressive flow-based causal discovery and inference"


This repository contains code to reproduce results presented in "Autoregressive flow-based causal 
discovery and inference", presented at the 2nd ICML workshop on Invertible Neural Networks, 
Normalizing Flows, and Explicit Likelihood Models (2020). 

The `main.py` script is the main gateway to reproduce the experiments details in the mansucript.
Run `python main.py -h` to learn about the arguments of the script:
```
usage: main.py [-h] [--dataset DATASET] [--nSims NSIMS]
               [--resultsDir RESULTSDIR] [-s] [-p] [-i] [-c]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset to run synthetic CD experiments on. Should be
                        either linear, hoyer2009 or nueralnet_l1 or all to run
                        all
  --nSims NSIMS         Number of synthetic simulations to run
  --resultsDir RESULTSDIR
                        Path for saving results.
  -s, --simulation      run the CD exp on synthetic data
  -p, --pairs           run Cause Effect Pairs experiments
  -i, --intervention    run intervention exp on toy example
  -c, --counterfactual  run counterfactual exp on toy example
```

___
To reproduce causal discovery simulations run: 
```
python main.py -s --dataset all 
```
This will run our proposed method, as well as baseline methods on the simulated data, as described in the manuscript.
Then it will plot Figure 1.

___
To run the proposed method on CauseEffectPair dataset, use:
```
python main.py -p
```
The percentage of correct inferred causal directions is printed to standard output,
and updated online after each new pair.
The values for baseline methods were taken from their respective papers.

___
To perform interventions using the proposed method on the toy example described in the manuscript, run:
```
python main.py -i
```

___
To perform counterfactuals using the proposed method on the toy example described in the manuscript, run:
```
python main.py -c
```


### Dependencies
This project was tested with the following versions:

- python 3.7
- numpy 1.18.2
- pytorch 1.4
- scikit-learn 0.22.2
- scipy 1.4.1
- matplotlib 3.2.1
- seaborn 0.10

This project uses normalizing flows implementations from [this](https://github.com/karpathy/pytorch-normalizing-flows) library. 

### Slurm usage

To run simulations in parallel:
```bash
for SIZE in 25 50 75 100 150 250 500; do
    for ALGO in CAReFl LRHyv notears RECI ANM; do
        for DSET in linear hoyer2009 nueralnet_l1; do
            sbatch slurm_main.sbatch -s --config simulations.yaml -m $DSET -a $ALGO -n $SIZE
        done
    done
done

```
___

To run interventions in parallel:
```bash
for SIZE in 500 750 1000 1250 1500 2000 2500; do
    for ALGO in carefl gp linear; do
        sbatch slurm_main.sbatch -i --config interventions.yaml -a $ALGO -n $SIZE
    done
done

```