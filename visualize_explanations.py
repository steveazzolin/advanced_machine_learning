import argparse
import torch
import os
from joblib import load as joblib_load
from collections import defaultdict

import Models.models_CORA as models_CORA
import Models.models_MUTAG as models_MUTAG
import Models.models_REDDIT as models_REDDIT
import Models.models_BAshapes as models_BAshapes

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


def read_explanations(path, divide_per_split=False):
    ret = defaultdict(lambda: defaultdict(list))
    for split in os.listdir(path):
        if ".txt" in split: continue
        for c in os.listdir(os.path.join(path, split)):
            if divide_per_split:
                key = split
            else:
                key = "all"
            for file in os.listdir(os.path.join(path, split, c)):
                tmp = joblib_load(os.path.join(path, split, c, file))
                data = tmp
                ret[key][c].append((data, file.split(".")[0]))
    return ret


def find_unique_explanations(expls, log=False):
    """
        Filter unique explanations based on isomorphism test
    """
    library = defaultdict(lambda: defaultdict(list))
    for split in expls.keys():
        for c in expls[split].keys():
            for graph in expls[split][c]:
                add = True
                for prev_elem in library[split][c]:
                    if nx.is_isomorphic(prev_elem[0], graph[0]):
                        add = False
                        break
                if add:
                    library[split][c].append(graph)
    if log:
        print("Before filtering:")
        for split in expls.keys():
            print(split)
            for c in expls[split].keys():
                print(f"\t {c}: {len(expls[split][c])}")
        print("After filtering:")
        for split in expls.keys():
            print(split)
            for c in expls[split].keys():
                print(f"\t {c}: {len(library[split][c])}")
    return library


def plot_single_graph(graph, ax):
    if nx.is_weighted(graph):
        edges , weights = zip(*nx.get_edge_attributes(graph, 'weight').items())

        # assign red color to the target node
        color_map = []
        for node in graph.nodes():
            if node == graph.graph["target_node"]:
                color_map.append("red")
            else:
                color_map.append("blue")
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, node_size=40, node_color=color_map, ax=ax, edge_color="blue", width=weights)
    elif len(graph.nodes()) > 0:
        nx.draw(graph)
        print(graph)
        raise NotImplementedError("Graph non-weighted")
    

def process_explanations(expls, cut_edges, cut_cc):
    """
        Process the local explanations such as to cut irrelavant edges and/or remove connected components not including the target node
    """
    ret = defaultdict(lambda: defaultdict(list))
    for split in expls.keys():
        for c in expls[split].keys():
            for graph , node_idx in expls[split][c]:
                edges , weights = zip(*nx.get_edge_attributes(graph,'weight').items())
                weights = [w+1 for w in weights]
                sorted_weights = sorted(weights, reverse=True)

                # define threshold to cut edges
                stop_i = np.mean(sorted_weights) # backup threshold
                for i in range(len(sorted_weights)-2):
                    if i <= 5: # include at least 5 edges
                        continue
                    if sorted_weights[i-1] - sorted_weights[i] >= 10 * (sorted_weights[i-2] - sorted_weights[i-1]) / 100 + (sorted_weights[i-2] - sorted_weights[i-1]):
                        stop_i = sorted_weights[i]
                        break

                # build a tmp graph to be plotted
                G_plot = nx.Graph(target_node=graph.graph["target_node"])
                for j , i in enumerate(weights):
                    if not cut_edges or i >= stop_i:  
                        G_plot.add_edge(edges[j][0], edges[j][1], weight=i)
                        G_plot.add_edge(edges[j][1], edges[j][0], weight=i)

                # remove connected components not including the target node
                if cut_cc:
                    for component in list(nx.connected_components(G_plot)):
                        tmp = list(component)
                        if graph.graph["target_node"] not in tmp:
                            for node in tmp:
                                G_plot.remove_node(node)
                if len(G_plot.nodes()) > 0:
                    ret[split][c].append((G_plot, node_idx))
        return ret

def plot_k_per_class(expls, labels, k):
    figsize = (10, 8)
    cols = k
    rows = len(expls["all"].keys())

    axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    for i in range(rows):
        for j , (elem , node_idx) in enumerate(expls["all"][str(i)][:cols]):
            axs[i, j].set_title(f"pred: {i} lbl: {labels[int(node_idx)]}")
            plot_single_graph(elem, axs[i,j])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to use for explanations.')
    parser.add_argument('--dataset', default="", help='Dataset to explain.')
    parser.add_argument('--expl', default="", help='Explainer to use.')
    parser.add_argument('--time', default="", help='Time of reference experiment.')
    parser.add_argument('--k', default=5, help='Number of plots per class.', type=int)
    parser.add_argument('--cut_edges', action='store_true', default=False, help='Whether to apply a threshold on edges with low score.')
    parser.add_argument('--cut_cc', action='store_true', default=False, help='Whether to cut the connected components not including the target node.')
    args = parser.parse_args()
    
    torch.manual_seed(42)

    pretrained_models = [f.split(".")[0] for f in os.listdir("Pretrained models") if os.path.isfile(os.path.join("Pretrained models", f))]
    explainers = ["subgraphx", "pgexplainer"]
    
    assert args.model + "_" + args.dataset in pretrained_models , "Model not yet implemented or trained"
    assert args.expl.lower() in explainers , "Explainer not yet implemented"

    if args.dataset.upper() == "CORA":
        fw = models_CORA.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "MUTAG":
        fw = models_MUTAG.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "REDDIT":
        fw = models_REDDIT.getFrameworkByName(args.model.upper())
    elif args.dataset.upper() == "BASHAPES":
        fw = models_BAshapes.getFrameworkByName(args.model.upper())

    path = f"Explanations/{args.expl}/{args.dataset}/{args.model}/{args.time}"

    expls = read_explanations(path, divide_per_split=False)
    expls = process_explanations(expls, cut_edges=args.cut_edges, cut_cc=args.cut_cc)
    expls = find_unique_explanations(expls, log=True)
    plot_k_per_class(expls, labels=fw.dataset.data.y, k=args.k)
