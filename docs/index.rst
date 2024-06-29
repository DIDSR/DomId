.. DomId documentation master file, created by
   sphinx-quickstart on Thu Dec  1 13:04:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DomId's documentation!
=================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:


About DomId
===============

DomId is a Python package offering a PyTorch-based suite of unsupervised deep clustering algorithms. The primary goal is to identify subgroups that have not been previously annotated within image datasets.

Some of the implemented models are the Variational Deep Embedding (VaDE) model [Jiang et al., 2017], Conditionally Decoded Variational Deep Embedding (CDVaDE) [Sidulova et al., 2023], Deep Embedding Clustering (DEC) [Xie et al., 2016], Structural Deep Clustering Network (SDCN) [Bo et al., 2020].

These clustering algorithms include a feature extractor component, which can be either an Autoencoders (AE) or a Variational Autoencoder (VAE). The package provides multiple AE and VAE architectures to choose from and includes instructions for extending the package with custom neural network architectures or clustering algorithms.

Experiment tutorials in Jupyter notebooks are available for both the MNIST dataset and a digital pathology dataset.

By adopting a highly modular design, the codebase prioritizes straightforward extensibility, so that new models, datasets or tasks can be added with ease.
The software design of DomId follows the design principles of [DomainLab](https://github.com/marrlab/DomainLab), which is a modular Python package for training domain invariant neural networks and has been used to develop DomId.

.. toctree::
   :maxdepth: 1
   :caption: Introduction and Quick Start guide:

   readme_link


.. toctree::
   :maxdepth: 1
   :caption: More information about the models:

   about_link


Loading a Datasets and Defining a Task
=======================================
Running the code with the custom dataset entails initialization of a Task and a Dataset. Examples of each for both HER2 and MNIST are included. More could be found in the sections below.

.. toctree::
   :maxdepth: 1
   :caption: Datasets and Tasks

   domid.dsets
   domid.tasks


Composition and Defining a Model
==================================

The detailed architecture of the linear and convolution encoders and decoders can be found in the compos directory, which served as the building blocks for the model. However, it's important to note that the specific clustering implementations of the models are defined in the domid/models directory. For further information, please refer to the details below...
As the pretraining, training, and performance evaluation metrics are specific to each model, they have been defined in their respective model modules.

.. toctree::
   :maxdepth: 2
   :caption: Models

   domid.compos
   domid.models


Training a Model
=================

The training process for the model is divided into three components:  Builder, Observer and Trainer.
Builder defines the model architecture and the task that is going to be used for the experiement.
The Observer is responsible for logging the training and validation losses and metrics, while the Trainer focuses on training the model itself.
The Observer defines the training and validation losses and metrics, while the Trainer saves the model and experiment results . For more information, please refer to the details provided bel0ow.

.. toctree::
   :maxdepth: 2
   :caption: Training a Model

   domid.algos
   domid.trainers


Ulitilies
=================
Some functions that are used for evaluation of the performance of the model are defined in the utils directory. For more information, please refer to the details provided below.

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   domid.utils



