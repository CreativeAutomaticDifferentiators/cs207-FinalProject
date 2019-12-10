from graphviz import Digraph
from OttoDiff.reverse import *

def createGraph(root, computationalGraph):
    # use dfs to traverse the computational graph
    if len(root.parents) == 0:
        return

    for par in root.parents:
        computationalGraph.edge(str(par), str(root))
        createGraph(par, computationalGraph)

def visualize(final_output_node):
    """Find the partial derivative of f with respect to given x

    INPUTS
    =======
    final_output_node: VariableNode of the function

    RETURNS
    ========
    Create the graph and store it in Digraph.gv.pdf in root directory

    EXAMPLES
    =========
    >>> x = VariableNode(3)
    >>> f = 2 * x + np.sin(x)
    >>> visualize(f)
    """
    computationalGraph = Digraph()
    createGraph(final_output_node, computationalGraph)
    computationalGraph.view()
