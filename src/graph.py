import networkx as nx
import torch as t
import torch_geometric as pyg
import gspan_mining as gspan

from graph_utils import edge_index_to_adj

def networkx_to_pyg_graph(graph):
    edge_index = t.tensor(list(graph.edges())).T
    try:
        return pyg.data.Data(
            x = t.tensor(list(graph.nodes(data = True))),
            edge_index = edge_index
        )
    except:
        # if the networkx nodes have no data, create constant 1 features:
        x = t.ones((len(graph.nodes), 1))
        return pyg.data.Data(x, edge_index)

def pyg_to_networkx_graph(graph):
    return pyg.utils.to_networkx(graph)

def adjacency_matrix_to_pyg_graph(A):
    return pyg.data.Data(
        x = t.arange(A.shape[0]),
        edge_index = t.tensor(A.nonzero())
    )

def pyg_to_adjacency_matrix(graph):
    return edge_index_to_adj(graph.edge_index)

def networkx_to_gspan(graph, multiplicity=1):
    # this format is described here: https://github.com/betterenvi/gSpan/tree/master/graphdata
    s = ""
    for n in range(multiplicity):
        s += f"t # {n}\n"
        for i, node in enumerate(graph.nodes(data = True)):
            node_label = 1 # node[1]["label"]
            # NB: node_labels are currently not used;
            #     I think they need to be discrete
            s += "v {} {}\n".format(i, node_label)
        for edge in graph.edges(data = True):
            edge_label = 1 # edge[2]["label"]
            s += "e {} {} {}\n".format(edge[0], edge[1], edge_label)
        if n == multiplicity - 1:
            s += "t # -1\n"
        # ^^ required to avoid a bug according to the gSpan docs ... :) :) :)
    return s

def gspan_to_pyg(str):
    rows = str.split("\n")
    rows = rows[1:-1] # throw away the "t # 0" header, and the last "t # -1" footer
    
    edge_index = []
    edge_attr = []
    x = []

    for line in rows:
        parts = line.strip().split(' ')
        if parts[0] == 'v':
            x.append([float(parts[2])])
        elif parts[0] == 'e':
            src, tgt, weight = int(parts[1]), int(parts[2]), float(parts[3])
            edge_index.append([src, tgt])
            edge_attr.append([weight])

    x = t.tensor(x, dtype=t.float)
    edge_index = t.tensor(edge_index, dtype=t.long).t().contiguous()
    edge_attr = t.tensor(edge_attr, dtype=t.float)
    return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def get_gspan_graph_str(gspan_graph):
    # Modification of https://github.com/betterenvi/gSpan/blob/master/gspan_mining/graph.py
    # to fix the bugs in it ... :)
    display_str = 't # {}\n'.format(gspan_graph.gid)
    for vid in gspan_graph.vertices:
        display_str += 'v {} {}\n'.format(vid, gspan_graph.vertices[vid].vlb)
    for frm in gspan_graph.vertices:
        edges = gspan_graph.vertices[frm].edges
        for to in edges:
            if gspan_graph.is_undirected:
                if frm < to:
                    display_str += 'e {} {} {}\n'.format(frm, to, edges[to].elb)
            else:
                display_str += 'e {} {} {}\n'.format(frm, to, edges[to].elb)
    display_str += 't # -1'
    return display_str
    

def subgraph(graph, node_set):
    """Returns the subgraph of graph induced by node_set.
    """
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
                    raise Exception(f"Edge index shape {edge_index.shape} not understood, must be [2, X]")
            self.pyg_data = pyg.data.Data(
                x = t.tensor(graph_data[0]),
                edge_index = edge_index
            )
        elif isinstance(graph_data, str):
            # assume it's a gspan string
            self.format = "gspan"
            # decode gspan:
            self.pyg_data = gspan_to_pyg(graph_data)
        elif hasattr(graph_data, "set_of_elb"):
            # asuming that no one else will name a graph property "set_of_elb" ...
            self.format = "gspan"
            self.pyg_data = gspan_to_pyg(get_gspan_graph_str(graph_data))
        else:
            raise ValueError("Unknown graph format")
        
        assert self.pyg_data.edge_index.shape[0] == 2, f"{self.pyg_data.edge_index.shape} is not a valid edge index shape, must be [2, X]"
    
    def to_networkx(self):
        return pyg_to_networkx_graph(self.pyg_data)
    def to_adjacency_matrix(self):
        return pyg_to_adjacency_matrix(self.pyg_data)
    def to_pyg(self):
        return self.pyg_data
    def to_gspan(self, multiplicity=1):
        return networkx_to_gspan(self.to_networkx(), multiplicity=multiplicity)
    
    def draw(self):
        nx.draw(self.to_networkx())
    
    def add_edge(self, node_a, node_b):
        self.pyg_data.edge_index = t.cat(
            (self.pyg_data.edge_index,
             t.tensor([[node_a], [node_b]])),
            dim = 1
        )
    
    def disjoint_union(self, graph_b):
        nx_union = nx.disjoint_union(self.to_networkx(), graph_b.to_networkx())
        return Graph(networkx_to_pyg_graph(nx_union))
    
    def num_nodes(self):
        return self.pyg_data.x.shape[0]
    
    def subgraph(self, node_set):
        return subgraph(self, node_set)
        

