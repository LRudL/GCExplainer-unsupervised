import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import sys

from sklearn import tree, linear_model
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.neighbors import NearestCentroid
import scipy.cluster.hierarchy as hierarchy

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, DenseDataLoader

from models import *

def load_syn_data(dataset_str):
    if dataset_str == "BA_Shapes":
        G = nx.readwrite.read_gpickle("../../data/BA_Shapes/graph_ba_300_80.gpickel")
        role_ids = np.load("../../data/BA_Shapes/role_ids_ba_300_80.npy")

    elif dataset_str == "Tree_Cycle":
        G = nx.readwrite.read_gpickle("../../data/Cycle_Shapes/graph_tree_8_60.gpickel")
        role_ids = np.load("../../data/Cycle_Shapes/role_ids_tree_8_60.npy")

    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids


def load_real_data(dataset_str):
    if dataset_str == "Mutagenicity":
        graphs = TUDataset(root='.', name='Mutagenicity')

    elif dataset_str == "Reddit_Binary":
        graphs = TUDataset(root='.', name='REDDIT-BINARY', transform=torch_geometric.transforms.Constant())
    else:
        raise Exception("Invalid Real Dataset Name")

    print()
    print(f'Dataset: {graphs}:')
    print('====================')
    print(f'Number of graphs: {len(graphs)}')
    print(f'Number of features: {graphs.num_features}')
    print(f'Number of classes: {graphs.num_classes}')

    data = graphs[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    return graphs


def prepare_syn_data(G, labels, train_split, if_adj=False):
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1)
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of edges: ", len(edges))

    return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

def prepare_real_data(graphs, train_split, batch_size):
    graphs = graphs.shuffle()

    train_idx = int(len(graphs) * train_split)
    train_set = graphs[:train_idx]
    test_set = graphs[train_idx:]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    full_loader = DataLoader(graphs, batch_size=len(graphs), shuffle=True)

    train_zeros = 0
    train_ones = 0
    for data in train_set:
        train_ones += np.sum(data.y.detach().numpy())
        train_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    test_zeros = 0
    test_ones = 0
    for data in test_set:
        test_ones += np.sum(data.y.detach().numpy())
        test_zeros += len(data.y.detach().numpy()) - np.sum(data.y.detach().numpy())

    print()
    print(f"Class split - Training 0: {train_zeros} 1:{train_ones}, Test 0: {test_zeros} 1: {test_ones}")


    return train_loader, test_loader, full_loader

def set_rc_params():
    small = 14
    medium = 18
    large = 24

    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=medium)
    plt.rc('figure', titlesize=large, figsize=(12, 8), facecolor='white')
    plt.rc('legend', loc='upper left')


def plot_activation_space(data, labels, activation_type, layer_num, path, note=""):
    rows = len(data)
    fig, ax = plt.subplots()
    ax.set_title(f"{activation_type} Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(labels)))

    plt.savefig(os.path.join(path, f"{layer_num}_layer.png"))
    plt.show()


def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note=""):
    fig, ax = plt.subplots()
    fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')

    for i in range(k):
        scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')

    ax.legend()
    plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
    plt.show()


def get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data=None):
    graphs = []
    color_maps = []
    labels = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
        # get neighbours
        neighbours = list()
        neighbours.append(idx)

        for i in range(0, num_expansions):
            new_neighbours = list()
            for e in edges:
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)

        color_map = []
        if graph_data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append('green')
                else:
                    color_map.append('pink')
        else:
            for node, attribute in zip(new_G, graph_data.x.numpy()):
                color_idx = np.argmax(attribute, axis=0)
                color_map.append(color_idx)

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])

    return graphs, color_maps, labels

def get_node_distances(clustering_model, data):
    if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
        x, y_predict = data
        clf = NearestCentroid()
        clf.fit(x, y_predict)
        centroids = clf.centroids_
        res = pairwise_distances(centroids, x)
        res_sorted = np.argsort(res, axis=-1)
    elif isinstance(clustering_model, KMeans):
        res_sorted = clustering_model.transform(data)

    return res_sorted


def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, num_nodes_view, edges, num_expansions, path, graph_data=None):
    res_sorted = get_node_distances(clustering_model, data)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(k, col, figsize=(20,20))
    fig.suptitle(f'Nearest to {clustering_type} Centroid for Layer {layer_num}', fontsize=40)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
            nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
            ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    views = ''.join((str(i) + "_") for i in num_nodes_view)
    if isinstance(clustering_model, AgglomerativeClustering):
        plt.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
    else:
        plt.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{views}view.png"))
    plt.show()

    return sample_graphs, sample_feat


def plot_dendrogram(data, reduction_type, layer_num, path):
    """Learned from: https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318 """
    fig, ax = plt.subplots()
    fig.suptitle(f'HC Dendrograms of {reduction_type} Activations of Layer {layer_num}')

    dendrogram = hierarchy.dendrogram(hierarchy.linkage(data, method='average'), truncate_mode="lastp", ax=ax, leaf_rotation=90.0, leaf_font_size=14)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Euclidean Distances")

    plt.savefig(os.path.join(path, f"hc_dendrograms_{reduction_type}.png"))
    plt.show()


def plot_completeness_table(model_type, calc_type, data, path):
    fig, ax = plt.subplots(figsize=(6, 3))
    headings = ["Model", "Data", "Completeness Score"]

    ax.set_title(f"Completeness Score (Task Accuracy) for {model_type} Models using {calc_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    calc_type = calc_type.replace(" ", "")
    plt.savefig(os.path.join(path, f"{model_type}_{calc_type}_completeness.png"))
    plt.show()


def calc_graph_similarity(top_graphs, max_nodes, num_nodes_view):
    top_G = top_graphs[0]

    print("Nodes ", top_G.number_of_nodes(), " Graphs ", len(top_graphs))

    if_iso = True

    for G in top_graphs[1:]:
        if not nx.is_isomorphic(top_G, G):
            if_iso = False
            break

    if if_iso:
        return 0

    if top_G.number_of_nodes() > max_nodes:
        return "skipping (too many nodes)"

    total_score = 0
    for G in top_graphs[1:]:

        if G.number_of_nodes() > max_nodes:
            return "skipping (too many nodes)"

        total_score += min(list(nx.optimize_graph_edit_distance(top_G, G)))

    return total_score / (len(top_graphs) - 1)


def plot_graph_similarity_table(model_type, data, path):
    fig, ax = plt.subplots(figsize=(25, 15))
    headings = ["Model", "Data", "Layer", "Concept/Cluster", "Graph Similarity Score"]

    ax.set_title(f"Graph Similarity for Concepts extracted using {model_type}")
    ax.axis('off')
    ax.table(cellText=data, colLabels=headings, loc="center", rowLoc="center", cellLoc="center", colLoc="center", fontsize=18)

    plt.savefig(os.path.join(path, f"{model_type}_graph_similarity.png"))
    plt.show()
