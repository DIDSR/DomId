# How to Add a New Dataset

This tutorial guides you through the steps to add a new dataset to our framework in a Python project.
Each dataset is loaded by the domain label, i.e., a separate dataset object is loaded for each domain. The corresponding domain labels are specified in the command line/experiment definition (`--tr_d`).

Within each such dataset (one per domain), individual observations have the form (X, Y), where X would be an image (or features, etc.) and Y would be its associated label. In general, the label Y is a separate concept from the domain label.

Follow the steps below to set up and configure your own dataset for the experiments.

## Step 1: Create Dataset File

1. Navigate to the `dset` folder in your project directory.
2. Create a new Python file named `dset_<name>.py`, replacing `<name>` with the name of your dataset.

## Step 2: Define the Dataset Class

Inside your new file, you will need to define a dataset class with necessary methods, along the lines of:

```python
class DsetYourClass:
    def __init__(self, domain_label):
        # Initialization code here


    def __getitem__(self, index):
        # Code to load an image and its associated labels
        # Returns:
        # - image: the loaded image
        # - image_label: the label associated with the image
        # - conditional_label: a condition associated with the image
        # - image_id: the identifier for the image
        return image, image_label, conditional_label, image_id
 ```

## Step 3: Implementing `__getitem__`

The `__getitem__` method is a key part of the dataset class, which should be implemented to load and return the necessary data:

- **image**: This should load the actual image from the dataset.
- **image_label**: Load the label (not the domain label) that is associated with the image.
- **conditional_label**: If your dataset includes conditions (for example, any additionally annotated label), this can be included using a CSV file.
- **image_id**: Useful for tracking which image is being processed, especially in debugging or complex data handling scenarios.

## Step 4: Create a task class that would utilize your Dset class

1. Navigate to the tasks folder in your project directory.
2. Create a task file specific to your dataset. This file will manage loading one domain at a time and will be passed to the experiment.
4. The task_<name>.py should contain the following functions:

```python
class NodeTaskTemplate(NodeTaskDictCluster):
    """Basic template for tasks where categories are considered 'domains'
    """

    @property
    def list_str_y(self):
        """
        This task has no conventional labels; categories are treated as domains.
        """
        return mk_dummy_label_list_str("label_prefix", number_of_domains)

    @property
    def isize(self):
        """
        :return: Image size object storing image channels, height, width.
        """
        return ImSize(channels, height, width)

    def get_list_domains(self):
        """
        Get list of domain names.
        :return: List of domain names.
        """
        return mk_dummy_label_list_str("domain_prefix", number_of_domains)

    def get_dset_by_domain(self, args, na_domain, split=False):
        """
        Get a dataset by domain.
        :param args: Command line arguments.
        :param na_domain: Domain name.
        :param split: Whether to perform a train/validation split.
        :return: Training dataset, Validation dataset.
        """
        ratio_split = float(args.split) if split else False
        trans = [transforms.Resize((desired_height, desired_width)), transforms.ToTensor()]
        ind_global = self.get_list_domains().index(na_domain)
        dset = DsetYourClass(domain=ind_global, args=args, list_transforms=trans)

        if ratio_split > 0:
            train_len = int(len(dset) * ratio_split)
            val_len = len(dset) - train_len
            train_set, val_set = random_split(dset, [train_len, val_len])
        else:
            train_set = dset
            val_set = dset
        return train_set, val_set
```

## Step 5: Add new Task Chain in `TaskChainNodeGetter` Class

After defining your task class, you will need to integrate it into the processing chain.
This is typically done in the `zoo_task.py` file where multiple tasks are chained together for sequential processing.

Here's how to add your new task to the chain:

1. Navigate to  `zoo_task.py` to the `TaskChainNodeGetter` class.
2. Add your `NodeTaskTemplate` to the existing chain as shown below: 

```python
chain = NodeTaskTemplate(succ=chain)
```

## Conclusion

With the dataset class set up in your `dset_<name>.py` file that is imported to `task_<name>.py`, your new dataset is ready to be integrated into your project.
