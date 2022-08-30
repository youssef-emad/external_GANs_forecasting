# GANs Forecasting Framework

Deep Learning Framework for training and evaluating time-series forecasting using Generative Adversarial Networks (GANs). The framework is built using `python3.9` and [PyTorch](https://pytorch.org) and it's mainly adopted from Conditional-GAN ([CGAN](https://arxiv.org/abs/1411.1784)) and Forecasting GAN ([ForGAN](https://arxiv.org/abs/1903.12549)).

The project also started as an extension of the official [torch implementation](https://git.opendfki.de/koochali/forgan/-/tree/master) of ForGAN but then expanded to better suit our use-case of oil production forecasting.

## Requirements
Before running any code, make sure you are using `python3.9` and that you have all the following requirements installed.

```
numpy==1.23.3
torch==1.11.0
torchvision==0.12.0
scikit-learn==0.24.2
pandas==1.4.2
tqdm==4.62.2
darker==1.4.2
isort==5.9
pre-commit=2.15.0
```
You can also install them directly through the `requirements.txt` file.
```
pip install -r requirements.txt
```

## Quick Start
To make sure the framework is set properly and working as expected, you can run the `train_gan.py` or `train.py` scripts which starts training sessions using GANs or stand-alone generator respectively.
```
python train_gan.py --config-path config_gan.py --output-dir experiments

OR

python train.py --config-path config.py --output-dir experiments
```

## Easy To Use
The main feature of this framework is the ability to train CGANs and stand-alone models for time-series forecasting without writing lots of code. For changing hyper-parameters and running different experiment, you only need to update a straight-forward configuration file (`config_gan.py` or `config.py`). The doesn't only make experimenting easier but it also helps keeping track of differences.

Also, each experiment's configuration is saved as a `JSON` file in the given output directory so we can easily compare between experiments' setups.

## Monitoring and Generator Saving
Any experiment can be monitored through `TensorBoard` session showing the training and validation metrics defined in the given configuration.

The generator is saved, in the given output directory, both regularly and based on `best_saving_criteria` in which a `metric` and comparison `mode` are defined.

## New Generators and Discriminators
You can directly add new classes for both models in the `generators.py` and `discriminators.py` files but make sure that any `generator` is expected to produce a continuous (`float`) outputs of shape `(batch_size, output_window_size)` and any `discriminator` is expected to produce a confidence output, of shape `(batch_size, 1)` representing wether the given sample is real or fake.

Also any new `generator` must inherit from `G_Base` to be able to use the `train()` method.

## New Evaluation Metrics
In order to use new evaluation metrics, you only need to define them in the `eval_metrics.py` and add them, with a representative name, to the `metrics_mapper` in the same file.



