# Adding a New Model to the Domid Python Package

This tutorial will guide you through the steps to add a new model file to the `models` submodule in the `domid` Python package.


## Step 1: Create the Model File and Define the Model Class

Navigate to the `models` directory in the `domid` codebase and create a file named `model_<name>.py`.
In this file, you will construct the new model, define loss optimization functions, and configure any necessary clustering layers.
The layers of the model are defined in the `compos` submodule.
Here, you can find already implemented fully-connected and convolutional VAEs (Variational AutoEncoders) and AEs (AutoEncoders).
These components can be used as building blocks for your model.
Create a class for your model by extending a base model class from `domid`.
Typically, models extend from a common base class such as `a_model_cluster.py`, which provides some of the default functionalities, and are wrapped within a `mk_model` method:

```python
def mk_model(parent_class=AModelCluster):
    class CustomModel(parent_class):
        def __init__(self, arg1, arg2, ...):
            super(CustomModel, self).__init__()
            # Model initialization and layer definitions
            self.model = model

        def _inference(self, x):
            # ...

        def infer_d_v_2(self, x, inject_domain):
            # ...

        def _cal_reconstruction_loss_helper(self, x,y):
            # ...

        # Implement any additional methods necessary for your model
        def _cal_loss_(self, x, y):
            # ...

    return CustomModel
```

## Step 2: Implement a trainer function if needed

When integrating your model into the `domid` package,
you have the option to utilize an existing trainer from the package or define a new trainer that caters
to the specific needs of your model. Below are details on both approaches.

### Using an Existing Trainer

`domid` includes several generic trainers that are designed to work with a variety of models.
For example, `trainer_cluster.py`, which is compatible with VaDE and DEC models.

### Defining a New Trainer

If the existing trainers do not meet the specific requirements of your model,
you may need to define a new trainer. This involves:

**Creating a Trainer Class:** Define a class in Python that encapsulates all 
aspects of training your model. This includes initializing the model, 
running the training loops, handling validation, and potentially testing.

```python
class CustomTrainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def tr_epoch(self, epoch_number):
        # runs one epoch of experiemnt for more details look at any other the existing trainers
```
