# Conditional Infilling GAN for Data Augmentation in Mammmograph Classification

Code for ["Conditional Infilling GAN for Data Augmentation in Mammogram Classification"](https://arxiv.org/abs/1807.08093).

Training Model

`python run.py # To train the model`

Validating Model

`python run.py --val # to generate samples from the GAN`

Synthesizing Data

`python run.py --syn # to generate synthetic labeled data`

Input data should be 256x256x1 size patches.
