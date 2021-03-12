# Introduction

This work was the homework 01 submission for 16-824: Visual Learning and Recognition. The assigned served to be the introduction to using PyTorch to build, train, and tune some common deep learning CNN models such as CaffeNet and ResNet.

# Prerequisites

If this work is being performed on a local or personal workstation, Anaconda should be installed on the workstation. In addition, Tensorboard should also be installed as there is code here that publishes training statistics onto Tensorboard. If this work is being performed on an AWS EC2 machine, you can run this code on a Deep Learning AMU Ubuntu 18.04 with a g4dn.xlarge GPU. Before running this code on AWS EC2, type the following command to turn on PyTorch:

~~~
source activate pytorch_p36
~~~

You may also need to install Tensorboard on the EC2 machine as well:

~~~
pip install tensorboard
~~~

# How to Run

## Training

Before running the .ipynb files to train the models, you must configure several things.

If you are running the 'q2_caffenet_pascal.ipynb' file, configure the Tensorboard writer output folder in the trainerCaffeNet.py is what you want it to be called. This output folder must be changed for each run of the .pynb file. Else you can delete the folder and rerun the training file.

If you are running any other .pynb file, configure the Tensorboard writer output folder in the trainer.py file. Similar to configuring the trainerCaffeNet.py file, the output foler must be changed at each run or the folder needs to be deleted and rerun again. 

Hyperparameters can be found in each respective .ipynb file as well as in the util.py file where the ARGS class is declared.

Restart and rerun the .ipynb files when the helper .py files are modified and saved. 

## Analysis

View the 'q5_analysis.ipynb' file for after-training analysis of the CaffeNet and ResNet models. Several examples of images and analysis are written in the file.