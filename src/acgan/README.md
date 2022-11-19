# ACGAN for MI-based BCI

A combination of GAN and classifier network to improve classification results.

## How to Run?

First, in cmd add `pwd` to `PYTHONPATH`:
```
export PYTHONPATH=${PWD}
```

Second, run `model.py` to apply k-fold cross validation on the model:
```
streamlit run src/acgan/model.py
```
