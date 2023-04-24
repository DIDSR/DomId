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

The goal of this Python package is to provide a PyTorch-based platform for deep unsupervised clustering and domain identification.

Currently implemented models include the Variational Deep Embedding (VaDE) model, Conditionally Decoded Variational Deep Embedding (CDVaDE), Deep Embedding Clustering (DEC). Other deep clustering models will be added in the future.
For additioonal information see the sections below.
For basic usage examples see: :doc:`readme_link`.


.. toctree::
   :maxdepth: 1
   :caption: More information about the models

   about_link


DomainLab
==============
DomainLab is a submodule that has been used to develop DomID, and it aims at learning domain invariant features by utilizing data from multiple domains so the learned feature can generalize to new unseen domains.

.. toctree::
   :maxdepth: 1
   :caption: DomainLab



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





