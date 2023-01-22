# Deep Learning Manager -- How To Use 

---

This document gives an overview how to use the deep learning (`dl_manager`).

---

## Preparing to run 

before the manager can be run, all its dependencies must be installed.
This can be done using 

```shell 
python -m pip install -r requirements.txt
```

Additionally, when one wants to use the `visualize` command,
the following repository must be cloned inside the `deep_learning` directory:

``` 
git clone https://github.com/jmaarleveld/visualkeras
```

---

## Running - Main Help Message 

The basic command to run the manager is using `python __main__.py`. When specifying the 
`-h` flag, the following output is printed:

```
sage: __main__.py [-h] {gui,list,hyperparams,generator-params,make-features,run,visualize,combination-strategies,run_analysis} ...

options:
  -h, --help            show this help message and exit

Sub-commands:
  {gui,list,hyperparams,generator-params,make-features,run,visualize,combination-strategies,run_analysis}
    gui                 Start an auxiliary analysis GUI.
    list                List options for various CLI options
    hyperparams         View hyper-parameters for a classifier
    generator-params    View parameters for a feature generator
    make-features       Generate a collection of features
    run                 Train a classifier and store the results
    visualize           Visualize a classifier
    combination-strategies
                        Give a comprehensive overview of all available model combination strategies.
    run_analysis        Analyze the results of deep learning runs
```

This output give a list of multiple possible commands one can run. 
Links to more explanation can be found below:

- gui - This command is a stub and currently not in a usable state.
- [list](docs/list.md) - A utility to elaborate on options for some parameters.
- [hyperparams](docs/hyperparams.md) - Get hyperparameters for a classifier 
- [generator-params](docs/generator-param.md) - Parameters for feature generators 
- [make-features](docs/make-features.md) - Generate Feature Vectors 
- [run](docs/run.md) - Train and test a classifier 
- visualize - Visualize a deep learning model (undocumented)
- [combination-strategies](docs/combination-strategies.md)
- [run_analysis](docs/analysis.md) - Analyze the results of the `run` command