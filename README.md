# Reproducibility

## How to run
```bash
cd dtu_mlops_3_docker_hydra
python vae_mnist.py
```

## Output folder
Outputs saved in output folder. This includes a folder for each run containing:
- Data visualisations (form Pytorch)
- log file (from logging module)
- Folder containing information about the run (from Hydra)


## Change params in cmdl
Dynamically changing and adding parameters on the fly from the command-line:

Change parameter(s):
```bash
python vae_mnist.py hyperparams_experiments=exp2
python vae_mnist.py hyperparams_experiments.seed=1234
```
The first line changes the used config file to exp2 while the second changes the parameter seed in the config file.

Add parameter(s):
```bash
python vae_mnist.py +hyperparams_experiments.stuff_that_i_want_to_add=42
```

## Test reproducability
Test reproducability:
```bash
python reproducibility_tester.py path/to/run/1 path/to/run/2
```