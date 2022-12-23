.. DomId documentation master file, created by
   sphinx-quickstart on Thu Dec  1 13:04:34 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DomId's documentation!
=================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

About DomID
===============
The Variational Deep Embedding (VaDE) model is trained to learn lower-dimensional representations of images based on a Mixture-of-Gaussians latent space prior distribution while optimizing cluster assignments. In this package, examples on multiple dataset has been presented.

.. toctree::
   :maxdepth: 1
   :caption: About

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

Model is built from the building blocks in domid/compos directory. However, the model for the experiment is defined in the domid/models. For more details, see below.

.. toctree::
   :maxdepth: 2
   :caption: Models

   domid.compos
   domid.models

Training a Model
=================
Training of the model consists of Observer and Trainer.
.. toctree::
   :maxdepth: 2
   :caption: Training a Model

   domid.algos
   domid.trainers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




