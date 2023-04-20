Datasets
==================

HER2 dataset module
-----------------------------
HER2 is a pathology dataset that contains 3 classes of breast cancer patches.
Note: to generate csv file with meta data for HER2 dataset, one can use 'generate_dataset_dataframe_her2.py' code.

.. automodule:: domid.dsets.dset_her2
   :members:
   :undoc-members:
   :show-inheritance:

MNIST dataset module
------------------------------
MNIST dataset is only used for CNN VaDE and Linear VaDE models. CDVaDE does not support MNIST due to the missing conditions.

.. automodule:: domid.dsets.dset_mnist
   :members:
   :undoc-members:
   :show-inheritance:

Colored MNIST dataset module
------------------------------
Colored MNIST is modified version of the colored MNIST task in DomainLab which allows for the specific numbers loaded.
Secondly, csv metadata file is created for colored MNIST digits, allowing for the conditioning in CDVaDE.

.. automodule:: domid.dsets.a_dset_mnist_color_rgb_solo
   :members:
   :undoc-members:
   :show-inheritance: