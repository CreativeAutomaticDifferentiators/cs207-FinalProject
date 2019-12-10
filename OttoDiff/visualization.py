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
    computationalGraph = Digraph()
    createGraph(final_output_node, computationalGraph)
    computationalGraph.view()


# x = VariableNode(3)
# y = VariableNode(5)
# z = VariableNode(7)
# # build the tree
# v = x / 2 + 2 / x + 2 * y + z
# print(find_df_dx(f=v, x=x))
#
# visualize(v)