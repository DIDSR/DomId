Datasets
==================

HER2 dataset module
-----------------------------
HER2 is a pathology dataset that contains 3 classes of breast cancer patches. This dataset is not publicly available (for more detail see: Keay et al., Journal of Pathology Informatics, 2013; Gavrielides et al., Archives of Pathology & Laboratory Medicine, 2011).
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
The Colored MNIST dataset is a modified version of the Colored MNIST task in DomainLab. It allows to specify a subset of the specific digits to be loaded.
In addition, a CSV metadata file is created for the Colored MNIST digits, allowing for the conditioning in our CDVaDE implementation.

.. automodule:: domid.dsets.a_dset_mnist_color_rgb_solo
   :members:
   :undoc-members:
   :show-inheritance:
