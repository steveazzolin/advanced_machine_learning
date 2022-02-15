import argparse
import torch
import os
from joblib import load as joblib_load
from collections import defaultdict

import Models.models_CORA as models_CORA
import Models.models_MUTAG as models_MUTAG
import Models.models_REDDIT as models_REDDIT
import Models.models_BAshapes as models_BAshapes
import Models.models_CITESEER as models_CITESEER

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_explanations(path, divide_per_split=False):
    ret = defaultdict(lambda: defaultdict(list))
    n = 0
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
                if data.number_of_nodes() == 0:
                    n += 1
                    continue
                ret[key][c].append((data, file.split(".")[0]))
    print(f"Encountered {n} empty explanations")
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
    

def preprocess_explanations(expls, cut_edges, cut_cc):
    """
        Pre-process the local explanations such as to cut irrelavant edges and/or remove connected components not including the target node
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


def get_last_experiment(path):
    return sorted([f for f in os.listdir(path)])[-1]







def avg_knn(g, edge_attribute="weight"):
    res = np.mean(list(dict(nx.average_degree_connectivity(g,weight=edge_attribute)).values()))
    return res
def avg_clust(g, edge_attribute="weight"):
    res = np.mean(list(dict(nx.clustering(g,weight=edge_attribute)).values()))
    return res
def avg_beet(g, edge_attribute="weight"):
    res = np.mean(list(dict(nx.betweenness_centrality(g,weight=edge_attribute)).values()))
    return res
def intensity(g, edge_attribute="weight"):
    w = list(dict(nx.get_edge_attributes(g,edge_attribute)).values())
    return np.prod(w)**(1/len(w))
def coherence(g, edge_attribute="weight"):
    w = list(dict(nx.get_edge_attributes(g,edge_attribute)).values())    
    i = intensity(g)
    res = (1/len(w)) * np.sum(w)
    return i/res
def avg_eig_cent(g, edge_attribute="weight"):
    res = np.mean(list(dict(nx.eigenvector_centrality(g,weight=edge_attribute)).values()))
    return res
def avg_closeness(g, edge_attribute="weight"):
    res = np.mean(list(dict(nx.closeness_centrality(g,distance=edge_attribute)).values()))
    return res
def get_embedding(g, metrics):    
    res = []
    for m in metrics:
        res.append(m(g))        
    return res



def extract_features_per_graph(expls, log=False):
    """
        Extract features for every graph
    """
    metrics = [avg_closeness, coherence, intensity, avg_beet, avg_clust, avg_knn, avg_eig_cent]
    ret = defaultdict(lambda: defaultdict(list))
    for split in expls.keys():
        for c in expls[split].keys():
            for graph , node_idx in expls[split][c]:
                ret[split][c].append(get_embedding(graph, metrics))
    return ret


def visualize_embeddings(embs):
    all_embds , classes = [] , []
    for split in embs.keys():
        for c in embs[split].keys():
            class_embs = embs[split][c]
            all_embds.extend(class_embs)
            classes.extend([c]*len(class_embs))

    print(all_embds[:3])
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(StandardScaler().fit_transform(all_embds))
    print("PCA explained variance: ", pca.explained_variance_ratio_)

    emb_2d = np.array(emb_2d)
    classes = np.array(classes)
    for c in np.unique(classes):
        plt.scatter(emb_2d[classes == c,0], emb_2d[classes == c,1], c=np.random.rand(len(classes[classes == c])), label=c)
    plt.legend()

    kmedoids = KMedoids(n_clusters=2, random_state=42).fit(emb_2d)
    
    plt.scatter(kmedoids.cluster_centers_[:,0], kmedoids.cluster_centers_[:,1], marker='x', c="red")
    plt.show()    
    #kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(emb)
    


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
    elif args.dataset.upper() == "CITESEER":
        fw = models_CITESEER.getFrameworkByName(args.model.upper())

    path = f"Explanations/{args.expl}/{args.dataset}/{args.model}/"
    if args.time == "":
        args.time = get_last_experiment(path)
        print("Reading: ", args.time)
    path += str(args.time)

    expls = read_explanations(path, divide_per_split=False)
    expls = preprocess_explanations(expls, cut_edges=args.cut_edges, cut_cc=args.cut_cc)

    # METHOD 1: Extract features
    embs = extract_features_per_graph(expls, log=True)
    visualize_embeddings(embs)



    #plot_k_per_class(expls, labels=fw.dataset.data.y, k=args.k)
