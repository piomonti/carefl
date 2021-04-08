# Code for "Causal Autoregressive Flow"

This repository contains code to run and reproduce experiments presented in [Causal Autoregressive Flows](https://arxiv.org/abs/2011.02268), presented at the 24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021).

The repository originally contained the code to reproduce results presented in [Autoregressive flow-based causal discovery and inference](https://arxiv.org/abs/2007.09390), presented at the 2nd ICML workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models (2020).
Switch to the `workshop` branch to access this version of the code.



## Dependencies
This project was tested with the following versions:

- python 3.7
- numpy 1.18.2
- pytorch 1.4
- scikit-learn 0.22.2
- scipy 1.4.1
- matplotlib 3.2.1
- seaborn 0.10

This project uses normalizing flows implementation from this [repository](https://github.com/karpathy/pytorch-normalizing-flows).

## Usage

The `main.py` script is the main gateway to reproduce the experiments detailed in the mansucript, and is straightforward
to use. Type `python main.py -h` to learn about the options.

Hyperparameters can be changed through the configuration files under `configs/`.
The `main.py` is setup to read the corresponding config file for each experiment, but this can be overwritten using the
`-y` or `--config` flag.

The results are saved under the `run/` folder. This can be changed using the `--run` flag.

Running the `main.py` script will only produce data for a single set of parameters, which are specified in the config file.
These parameters include the dataset type, the number of simulations, the algorithm, the number of observations, the architectural parameters for the neural networks (number of layers, dimension of the hidden layer...), etc...

To reproduce the figures in the manuscript, the script should be run multiple time for each different combination of parameters, to generate the data used for the plots.
Convience scripts are provided to do this in parallel using SLURM (see below). These make use of certain debugging flags that overwrite certain fields in the config file.

Finally, the `flow.scale` field in the config files is used to switch from CAREFL to CAREFL-NS by setting it to `false`.


### Examples
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


## Reference

If you find this code helpful/inspiring for your research, we would be grateful if you cite the following:

```bib
@inproceedings{khemakhem2021causal,
  title = { Causal Autoregressive Flows },
  author = {Khemakhem, Ilyes and Monti, Ricardo and Leech, Robert and Hyvarinen, Aapo},
  booktitle = {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages = {3520--3528},
  year = {2021},
  editor = {Banerjee, Arindam and Fukumizu, Kenji},
  volume = {130},
  series = {Proceedings of Machine Learning Research},
  month = {13--15 Apr},
  publisher = {PMLR}
}
```


## License
A full copy of the license can be found [here](LICENSE).

```
MIT License

Copyright (c) 2020 Ilyes Khemakhem and Ricardo Pio Monti

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


