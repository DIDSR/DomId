Trainers
======================


Pretraining
---------------------------------------
For the VaDE, CDVaDE, and DEC models, pretraining involves initializing the models with an MSE loss function during the first few epochs.
In addition, a Gaussian mixture model is used to initialize the clusters.

.. automodule:: domid.trainers.pretraining_vade
   :members:
   :undoc-members:
   :show-inheritance:

Trainer (one epoch)
-----------------------------------
The trainer is designed to run for a single epoch of the training process, during which the model's loss is computed, the neural network's weights are updated, and the outcomes are recorded.

.. automodule:: domid.trainers.trainer_vade
   :members:
   :undoc-members:
   :show-inheritance:

