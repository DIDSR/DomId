"""
chain of responsibility pattern for algorithm selection
"""
from domainlab.utils.u_import import import_path

from domid.algos.builder_ae import NodeAlgoBuilderAE
from domid.algos.builder_dec import NodeAlgoBuilderDEC
from domid.algos.builder_m2yd import NodeAlgoBuilderM2YD
from domid.algos.builder_sdcn import NodeAlgoBuilderSDCN
from domid.algos.builder_vade import NodeAlgoBuilderVaDE


class AlgoBuilderChainNodeGetter:

    """
    1. Hardcoded chain
    3. Return selected node
    """

    def __init__(self, model, apath):
        self.model = model
        self.apath = apath
        #
        self._list_str_model = model.split("_")
        self.model = self._list_str_model.pop(0)

    def register_external_node(self, chain):
        """
        if the user specify an external python file to implement the algorithm
        """
        if self.apath is None:
            return chain
        node_module = import_path(self.apath)
        node_fun = node_module.get_node_na()
        newchain = node_fun(chain)
        return newchain

    def __call__(self):
        """
        1. construct the chain, filter out responsible node, create heavy-weight business object
        2. hard code seems to be the best solution
        """

        chain = NodeAlgoBuilderSDCN(None)
        chain = NodeAlgoBuilderDEC(chain)
        chain = NodeAlgoBuilderVaDE(chain)
        chain = NodeAlgoBuilderAE(chain)
        chain = NodeAlgoBuilderM2YD(chain)
        chain = self.register_external_node(chain)
        node = chain.handle(self.model)
        head = node
        while self._list_str_model:
            self.model = self._list_str_model.pop(0)
            node2decorate = self.__call__()
            head.extend(node2decorate)
            head = node2decorate
        return node
