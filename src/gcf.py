import networkx as nx
import torch as t
import torch_geometric as pyg
import gspan_mining
from gspan_mining.config import parser as gspan_parser
from gspan_mining.main import main as gspan_main
from graph import Graph, subgraph, graphs_to_gspan

# WARNING: UNTESTED CODE


def concept_purity(graph, concept):
    """Returns the graph_edit_distance of all graphs in the concept.
    Recall that each concept is a set of nodes. In this case, it is
    a set of node indices in the graph.
    """
    pairs = 0
    sum_ged = 0
    for node_set_a in concept:
        for node_set_b in concept:
            ged = nx.graph_edit_distance(
                subgraph(graph, node_set_a), subgraph(graph, node_set_b)
            )
            sum_ged += ged
            pairs += 1
    return sum_ged / pairs


class GraphConceptFinder:
    def __init__(self, graph_to_concepts_fn):
        self.graph_to_concepts_fn = graph_to_concepts_fn
        self.concepts = None  # the last call to find_concepts is stored here
        self.graph = None  # store the last graph passed to find_concepts

    def find_concepts(self, graph):
        concepts = self.graph_to_concepts_fn(graph)
        self.concepts = concepts
        # each concept is a set of nodes in the graph
        return concepts

    def save_concepts(self, graph, file):
        concepts = self.find_concepts(graph)
        # create file if it doesn't exist, overwrite if it does:
        t.save(concepts, file)
        return concepts

    def load_concepts(self, file):
        concepts = t.load(file)
        self.concepts = concepts
        return concepts

    def concept_purities(self):
        purities = []
        for concept in self.concepts:
            purities.append(concept_purity(self.graph, concept))
        return purities


# gspan_concept_finder = GraphConceptFinder(lambda graph: gspan(graph.to_gspan()))

def gspan_on_graph(graph, n=100, k=5, support=2, temp_location="../gspan_out"):
    # k is the subgraph size
    
    subgraphs = []
    
    for _ in range(n):
        subgraphs.append(graph.random_subgraph(k))
    
    gspan_graphs_str = graphs_to_gspan(subgraphs)
    
    # write to file:
    with open(temp_location, "w") as file:
        file.truncate(0)
        file.write(gspan_graphs_str)
    args_str = f'-s {support} -d False -l 3 -u 6 -v False {temp_location}'
    print(args_str)
    FLAGS, _ = gspan_parser.parse_known_args(args=args_str.split())
    print(FLAGS)
    gs = gspan_main(FLAGS)
    # print(f"min sup: {gs._min_support}")
    return gs
    

