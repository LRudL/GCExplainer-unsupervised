import networkx as nx
import torch as t
import torch_geometric as pyg
import gspan_mining as gspan
import random
from utils import vector_hash_color

from graph_utils import edge_index_to_adj


def networkx_to_pyg_graph(graph):
    # if not nx.is_directed(graph):
    #     graph = graph.to_directed()
    # # edge_index = t.tensor(list(graph.edges())).T

    # # PROBLEM with above: graphs where nodes are not numbered in ascending order up from zero.
    # # Therefore:
    # # Create a mapping from the original node numbers to PyTorch node numbers
    # mapping = {n: i for i, n in enumerate(sorted(graph.nodes()))}
    # # Create the PyTorch Geometric data object
    # edge_index = t.tensor([[mapping[u], mapping[v]] for u, v in graph.edges()]).T

    # try:
    #     return pyg.data.Data(
    #         x=t.tensor(list(graph.nodes(data=True))), edge_index=edge_index
    #     )

    edge_index = t.tensor(list(graph.edges())).T
    try:
        return pyg.data.Data(
            x=t.tensor(list(graph.nodes(data=True))), edge_index=edge_index
        )
    except:
        # if the networkx nodes have no data, create constant 1 features:
        x = t.ones((len(graph.nodes), 1))
        return pyg.data.Data(x, edge_index)


def pyg_to_networkx_graph(graph):
    return pyg.utils.to_networkx(graph).to_undirected()


def adjacency_matrix_to_pyg_graph(A):
    return pyg.data.Data(x=t.arange(A.shape[0]), edge_index=t.tensor(A.nonzero()))


def pyg_to_adjacency_matrix(graph):
    return edge_index_to_adj(graph.edge_index)


def networkx_to_gspan(graph, multiplicity=1):
    # this format is described here: https://github.com/betterenvi/gSpan/tree/master/graphdata
    s = ""
    for n in range(multiplicity):
        s += f"t # {n}\n"
        for i, node in enumerate(graph.nodes(data=True)):
            node_label = 1  # node[1]["label"]
            # NB: node_labels are currently not used;
            #     I think they need to be discrete
            s += "v {} {}\n".format(i, node_label)
        for edge in graph.edges(data=True):
            edge_label = 1  # edge[2]["label"]
            s += "e {} {} {}\n".format(edge[0], edge[1], edge_label)
        if n == multiplicity - 1:
            s += "t # -1\n"
        # ^^ required to avoid a bug according to the gSpan docs ... :) :) :)
    return s


def graphs_to_gspan(graphs):
    # see also networkx_to_gspan
    s = ""
    for n, graph in enumerate(graphs):
        graph = graph.to_networkx()
        s += f"t # {n}\n"
        for i, node in enumerate(graph.nodes(data=True)):
            node_label = 1  # node[1]["label"]
            # NB: node_labels are currently not used;
            #     I think they need to be discrete
            s += "v {} {}\n".format(i, node_label)
        for edge in graph.edges(data=True):
            edge_label = 1  # edge[2]["label"]
            s += "e {} {} {}\n".format(edge[0], edge[1], edge_label)
        if n == len(graphs) - 1:
            s += "t # -1\n"
        # ^^ required to avoid a bug according to the gSpan docs ... :) :) :)
    return s


def _linebreak_gspan_str(gstr):
    # gstr looks like 'v 0 1 v 1 1 v 2 1 v 3 1 e 0 1 1 e 1 2 1 e 2 3 1 '
    # and needs to be 'v 0 1\nv 1 1\nv 2 1[...]'
    vstr, *estr = gstr.split("e")
    estr = "e" + "e".join(estr)
    sv = vstr.split(" ")
    se = estr.split(" ")
    s = "\n".join(
        list(
            map(
                lambda vij: " ".join(vij),
                [sv[i : i + 3] for i in range(0, len(sv) - len(sv) % 3, 3)],
            )
        )
        + list(
            map(
                lambda eabc: " ".join(eabc),
                [se[i : i + 4] for i in range(0, len(se) - len(se) % 4, 4)],
            )
        )
    )
    return "t # 0\n" + s + "\nt # -1\n"


def gspan_to_pyg(gstr):
    if gstr[0] != "t":
        # then this is the other type of gspan string
        # see _linebreak_gspan_str,
        # note that we need to fix this gstr because it is missing the line breaks:
        gstr = _linebreak_gspan_str(gstr)

    rows = gstr.split("\n")
    rows = rows[1:-1]  # throw away the "t # 0" header, and the last "t # -1" footer

    edge_index = []
    edge_attr = []
    x = []

    node_ids = []

    for line in rows:
        parts = line.strip().split(" ")
        if parts[0] == "v":
            node_ids.append(int(parts[1]))
            x.append([float(parts[2])])
        elif parts[0] == "e":
            src, tgt, weight = int(parts[1]), int(parts[2]), float(parts[3])
            edge_index.append([src, tgt])
            edge_attr.append([weight])

    # PROBLEM with above: gspan strings don't necessarily have vertices numbered
    # in ascending order up from zero.
    # Therefore:
    # Create a mapping from the original node numbers to PyTorch node numbers
    # mapping = {n: i for i, n in enumerate(sorted(node_ids))}
    # # Create the PyTorch Geometric data object
    # for i in range(len(edge_index)):
    #     for j in range(len(edge_index[i])):
    #         edge_index[i][j] = mapping[edge_index[i][j]]

    x = t.tensor(x, dtype=t.float)
    edge_index = t.tensor(edge_index, dtype=t.long).t().contiguous()
    edge_attr = t.tensor(edge_attr, dtype=t.float)
    return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_gspan_graph_str(gspan_graph):
    # Modification of https://github.com/betterenvi/gSpan/blob/master/gspan_mining/graph.py
    # to fix the bugs in it ... :)
    display_str = "t # {}\n".format(gspan_graph.gid)
    for vid in gspan_graph.vertices:
        display_str += "v {} {}\n".format(vid, gspan_graph.vertices[vid].vlb)
    for frm in gspan_graph.vertices:
        edges = gspan_graph.vertices[frm].edges
        for to in edges:
            if gspan_graph.is_undirected:
                if frm < to:
                    display_str += "e {} {} {}\n".format(frm, to, edges[to].elb)
            else:
                display_str += "e {} {} {}\n".format(frm, to, edges[to].elb)
    display_str += "t # -1"
    return display_str


def subgraph(graph, node_set):
    """Returns the subgraph of graph induced by node_set."""
    return Graph(graph.to_networkx().subgraph(node_set))


class Graph:
    def __init__(self, graph_data):
        ## figure out which format graph data is in:
        self.format = "unknown"
        if isinstance(graph_data, nx.Graph):
            self.format = "networkx"
            self.pyg_data = networkx_to_pyg_graph(graph_data)
        elif isinstance(graph_data, pyg.data.Data):
            self.format = "pyg"
            self.pyg_data = graph_data
        elif isinstance(graph_data, t.Tensor):
            self.format = "adjacency_matrix"
            self.pyg_data = adjacency_matrix_to_pyg_graph(graph_data)
        elif isinstance(graph_data, tuple):
            # assume it's a tuple of (node_labels, edges)
            self.format = "labels_and_edges"
            edge_index = t.tensor(graph_data[1])
            if edge_index.shape[0] != 2:
                if edge_index.shape[1] == 2:
                    print("WARNING: edge_index shape wrong way around")
                    edge_index = edge_index.T
                else:
                    raise Exception(
                        f"Edge index shape {edge_index.shape} not understood, must be [2, X]"
                    )
            self.pyg_data = pyg.data.Data(
                x=t.tensor(graph_data[0]), edge_index=edge_index
            )
        elif isinstance(graph_data, str):
            # assume it's a gspan string
            self.format = "gspan"
            self.pyg_data = gspan_to_pyg(graph_data)
        elif hasattr(graph_data, "set_of_elb"):
            # asuming that no one else will name a graph property "set_of_elb" ...
            self.format = "gspan"
            self.pyg_data = gspan_to_pyg(get_gspan_graph_str(graph_data))
        else:
            raise ValueError("Unknown graph format")

        assert (
            self.pyg_data.edge_index.shape[0] == 2
        ), f"{self.pyg_data.edge_index.shape} is not a valid edge index shape, must be [2, X]"

    def to_networkx(self):
        return pyg_to_networkx_graph(self.pyg_data)

    def to_adjacency_matrix(self):
        return pyg_to_adjacency_matrix(self.pyg_data)

    def to_pyg(self):
        return self.pyg_data

    def to_gspan(self, multiplicity=1):
        return networkx_to_gspan(
            self.to_networkx().to_undirected(), multiplicity=multiplicity
        )

    def draw(self):
        # use draw_around_node to draw a neighbourhood
        nx.draw(
            self.to_networkx(),
            node_size=50,
            labels={i: i for i in range(self.num_nodes())},
        )

    def add_edge(self, node_a, node_b):
        self.pyg_data.edge_index = t.cat(
            (self.pyg_data.edge_index, t.tensor([[node_a], [node_b]])), dim=1
        )

    def disjoint_union(self, graph_b):
        nx_union = nx.disjoint_union(self.to_networkx(), graph_b.to_networkx())
        return Graph(networkx_to_pyg_graph(nx_union))

    def num_nodes(self):
        return self.pyg_data.x.shape[0]

    def num_edges(self):
        return self.pyg_data.edge_index.shape[1]

    def __repr__(self):
        return "<Graph object with {} nodes and {} edges>".format(
            self.num_nodes(), self.num_edges()
        )

    def diameter(self):
        return nx.diameter(self.to_networkx())

    def subgraph(self, node_set):
        return subgraph(self, node_set)

    def subgraph_around_node(self, node, steps=2, nx_return=False):
        nxg = self.to_networkx()
        subgraph = nx.ego_graph(nxg, node, steps, undirected=False)
        if nx_return:
            return subgraph
        return Graph(subgraph)

    def draw_around_node(self, node, steps=2, ax=None, colourmap=None, labelmap=None):
        nxsg = self.subgraph_around_node(node, steps=steps, nx_return=True)
        if colourmap is None:
            colors = [
                vector_hash_color(self.pyg_data.x[node_i]) for node_i in nxsg.nodes
            ]
        else:
            colors = [colourmap(self.pyg_data.x[node_i]) for node_i in nxsg.nodes]

        if labelmap is None:
            labels = {i: i if i != node else f"[{i}]" for i in nxsg.nodes}
        else:
            labels = {i: labelmap(self.pyg_data.x[i]) for i in nxsg.nodes}

        nx.draw(
            nxsg,
            node_size=200,
            labels=labels,
            node_color=colors,
            ax=ax,
            alpha=0.75,
        )

    def random_subgraph(self, k=6):
        # k is the number of steps from a starting node to take to form the subgraph
        nxg = self.to_networkx().to_undirected()
        node_choice = random.choice(list(nxg.nodes()))
        # print(node_choice)
        subgraph = nx.ego_graph(nxg, node_choice, k, undirected=True)

        # NEVERMIND, the below was a consequence of a bug in networkx_to_pyg_graph
        # for some reason it will include spurious nodes so:
        # isolated_nodes = list(nx.isolates(subgraph))
        # print(f"isolated nodes: {isolated_nodes}")
        # print(f"nodes: {subgraph.nodes()}")
        # print(f"edges: {subgraph.edges()}")
        # for node in isolated_nodes:
        #     print(f"removing node {node}")
        #     subgraph.remove_node(node)

        return Graph(subgraph)
