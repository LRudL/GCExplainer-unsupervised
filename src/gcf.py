import networkx as nx
import torch as t
import torch_geometric as pyg
import gspan_mining
from gspan_mining.config import parser as gspan_parser
from gspan_mining.main import main as gspan_main
from graph import Graph, subgraph, graphs_to_gspan
import pandas as pd
import matplotlib.pyplot as plt

# WARNING: UNTESTED CODE


def concept_purity(graph, concept, max_num=50):
    """Returns the graph_edit_distance of all graphs in the concept.
    Recall that each concept is a set of node centres, each defining
    a step-2 subgraph.
    """
    pairs = 0
    sum_ged = 0
    num_processed = 0
    for node_center_a in concept:
        num_processed += 1
        if num_processed > max_num:
            print("WARNING: concept_purity exceeded max_num")
            break
        for node_center_b in concept:
            subgraph_a = graph.subgraph_around_node(node_center_a, steps=2)
            subgraph_b = graph.subgraph_around_node(node_center_b, steps=2)
            ged = nx.graph_edit_distance(
                subgraph_a.to_networkx(), subgraph_b.to_networkx()
            )
            sum_ged += ged
            pairs += 1
    return sum_ged / pairs


class GraphConceptFinder:
    def __init__(self, graph_to_concepts_fn):
        self.graph_to_concepts_fn = graph_to_concepts_fn
        self.concepts = None  # the last call to find_concepts is stored here
        self.graph = None  # store the last graph passed to find_concepts
        # self.verbose = False
        self.concept_representation = "centers"

    def find_concepts(self, graph):
        concepts = self.graph_to_concepts_fn(graph)
        self.concepts = concepts
        if isinstance(concepts[0][0], int):
            self.concept_representation = "centers"
        else:
            # assume concepts are already full subgraphs
            self.concept_representation = "subgraphs"
        self.graph = graph
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

    def concept_purities(self, max_num=50):
        purities = []
        for concept in self.concepts:
            purities.append(concept_purity(self.graph, concept, max_num=max_num))
        return purities

    def draw_concepts(self, num_examples=5, colourmap=None, labelmap=None):
        fig, axs = plt.subplots(len(self.concepts), num_examples, figsize=(20, 20))
        print(
            "Each row is a concept. Each item in a row is an example of that concept."
        )
        for i, concept in enumerate(self.concepts):
            for j in range(num_examples):
                if j >= len(concept):
                    axs[i][j].axis("off")
                    break
                print(concept[j])
                self.graph.draw_around_node(
                    concept[j],
                    steps=2,
                    ax=axs[i][j],
                    colourmap=colourmap,
                    labelmap=labelmap,
                )
        fig.show()


# gspan_concept_finder = GraphConceptFinder(lambda graph: gspan(graph.to_gspan()))


def gspan_results(
    graph,
    n=100,  # the number of subgraphs to sample
    k=5,  # k is the subgraph size
    support=30,
    min_nodes=4,
    max_nodes=10,
    temp_location="../gspan_out",
):

    subgraphs = []

    for _ in range(n):
        subgraphs.append(graph.random_subgraph(k))

    gspan_graphs_str = graphs_to_gspan(subgraphs)

    # write to file:
    with open(temp_location, "w") as file:
        file.truncate(0)
        file.write(gspan_graphs_str)

    args_str = f"-s {support} -d False -l {min_nodes} -u {max_nodes} {temp_location}"
    print(args_str)
    FLAGS, _ = gspan_parser.parse_known_args(args=args_str.split())

    # print(FLAGS)
    gs = gspan_main(FLAGS)
    # print(f"min sup: {gs._min_support}")

    df = gs._report_df.sort_values(by=["support"], ascending=False)

    diameters = pd.Series(
        [Graph(df.iloc[i]["description"]).diameter() for i in range(len(df))]
    )

    df["score"] = [
        df.iloc[i]["support"] * df.iloc[i]["num_vert"] / diameters.iloc[i]
        for i in range(len(df))
    ]

    df = df.sort_values(by=["score"], ascending=False)

    return df


def gspan_on_graph(
    graph,
    concepts=7,  # the number of concepts to return
    n=100,  # the number of subgraphs to sample
    k=5,  # k is the subgraph size
    support=2,
    min_nodes=4,
    max_nodes=10,
    temp_location="../gspan_out",
):
    df = gspan_results(
        graph,
        n=n,
        k=k,
        support=support,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        temp_location=temp_location,
    )

    concept_graphs = []
    for i in range(concepts):
        gstr = df.iloc[i]["description"]
        concept = Graph(gstr)
        concept_graphs.append(concept)

    return concept_graphs, df
