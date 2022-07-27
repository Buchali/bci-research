# Conditional W-GAN for MI-based BCI

Generating artificial data to improve the classification of MI tasks based on 2D stft representations of EEG signals.

## How to Run?

First, in cmd add `pwd` to `PYTHONPATH`:
```
export PYTHONPATH=${PWD}
```

Second, run `training.py` to train the generator and critic networks:
```
python src/wgan/training.py
```
