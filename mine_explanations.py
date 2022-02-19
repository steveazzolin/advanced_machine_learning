import argparse
from async_timeout import timeout
import torch
import os
from joblib import load as joblib_load
from collections import defaultdict
from tqdm import tqdm

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

NUM_EXPLS = 0
def read_explanations(path, divide_per_split=False, labels=None, splits=None):
    global NUM_EXPLS
    ret = defaultdict(lambda: defaultdict(list))
    n = 0
    for split in os.listdir(path):
        if ".txt" in split: continue
        for c in os.listdir(os.path.join(path, split)):
            if divide_per_split:
                if split not in splits:
                    continue
                key = split
            else:
                key = "all"
            for file in os.listdir(os.path.join(path, split, c)):
                tmp = joblib_load(os.path.join(path, split, c, file))
                node_idx = int(file.split(".")[0])
                data = tmp
                if data.number_of_nodes() == 0 or (labels is not None and labels[node_idx] != int(c)):
                    n += 1
                    continue
                ret[key][c].append((data, file.split(".")[0]))
                NUM_EXPLS += 1
    print(f"Encountered {n} empty/wrong explanations")
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
        weights = [(w)**2 for w in weights]

        # assign red color to the target node
        color_map = []
        for node in graph.nodes():
            if node == graph.graph["target_node"]:
                color_map.append("red")
            else:
                color_map.append("blue")
        pos = nx.spring_layout(graph, seed=42)
        s = nx.draw(graph, pos, node_size=40, node_color=color_map, ax=ax, edge_color="blue", width=weights)
    elif len(graph.nodes()) > 0:
        nx.draw(graph)
        print(graph)
        raise NotImplementedError("Graph non-weighted")
    

def elbow_method(weights,):
    sorted_weights = sorted(weights, reverse=True)

    # define threshold to cut edges
    stop = 0.4e-6 #np.mean(sorted_weights) # backup threshold
    for i in range(len(sorted_weights)-2):
        if i < 5: # include at least 5 edges
            continue
        if sorted_weights[i-1] - sorted_weights[i] >= 10 * (sorted_weights[i-2] - sorted_weights[i-1]) / 100 + (sorted_weights[i-2] - sorted_weights[i-1]):
            stop = sorted_weights[i]
            break
    return stop

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
                
                stop = elbow_method(weights)

                # build a tmp graph to be plotted
                G_plot = nx.Graph(target_node=graph.graph["target_node"])
                for j , i in enumerate(weights):
                    if not cut_edges or i >= stop:  
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
    rows = len(expls["train"].keys())

    axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    for i in range(rows):
        for j , (elem , node_idx) in enumerate(expls["train"][str(i)][:cols]):
            axs[i, j].set_title(f"lbl: {labels[int(node_idx)]} diam: {diameter(elem)}")
            plot_single_graph(elem, axs[i,j])
    plt.show()

def plot_edge_weight_distribution(expls, k):
    figsize = (10, 8)
    cols = k
    rows = len(expls["train"].keys())

    axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    for i in range(rows):
        for j , (graph , node_idx) in enumerate(expls["train"][str(i)][:cols]):
            edges , weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
            weights = sorted(weights)            
            cut_point = elbow_method(weights)
            
            #axs[i, j].set_title(f"backup cut")
            axs[i, j].plot(list(range(len(weights))), weights)
            axs[i, j].plot([0, len(weights)], [cut_point, cut_point])
            axs[i, j].plot([len(weights) - 5, len(weights) - 5], [0, weights[-1]])
    plt.show()


def get_last_experiment(path):
    return sorted([f for f in os.listdir(path)])[-1]





#  _______  _______ _________          _______  ______     __   
# (       )(  ____ \\__   __/|\     /|(  ___  )(  __  \   /  \  
# | () () || (    \/   ) (   | )   ( || (   ) || (  \  )  \/) ) 
# | || || || (__       | |   | (___) || |   | || |   ) |    | | 
# | |(_)| ||  __)      | |   |  ___  || |   | || |   | |    | | 
# | |   | || (         | |   | (   ) || |   | || |   ) |    | | 
# | )   ( || (____/\   | |   | )   ( || (___) || (__/  )  __) (_
# |/     \|(_______/   )_(   |/     \|(_______)(______/   \____/ 

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
def diameter(g):
    return nx.algorithms.distance_measures.diameter(g)
def get_embedding(g, metrics):    
    res = []
    for m in metrics:
        res.append(m(g))        
    return res

def extract_features_per_graph(expls, log=False):
    """
        Extract features for every graph
    """
    with tqdm(total=NUM_EXPLS) as pbar:
        metrics = [avg_closeness, coherence, intensity, avg_beet, avg_clust, avg_knn, diameter, avg_eig_cent]
        ret = defaultdict(lambda: defaultdict(list))
        for split in expls.keys():
            for c in expls[split].keys():
                for graph , node_idx in expls[split][c]:
                    ret[split][c].append(get_embedding(graph, metrics))
                    pbar.update(1)
    return ret


def visualize_embeddings(embs, k):
    all_embds , classes = [] , []
    for split in embs.keys():
        for c in embs[split].keys():
            class_embs = embs[split][c]
            all_embds.extend(class_embs)
            classes.extend([c]*len(class_embs))

    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(StandardScaler().fit_transform(all_embds))
    print("PCA explained variance: ", pca.explained_variance_ratio_)

    emb_2d = np.array(emb_2d)
    classes = np.array(classes)
    for c in np.unique(classes):
        plt.scatter(emb_2d[classes == c,0], emb_2d[classes == c,1], label=c)
    plt.legend()

    #kmedoids = KMedoids(n_clusters=k, random_state=42).fit(emb_2d)    
    #plt.scatter(kmedoids.cluster_centers_[:,0], kmedoids.cluster_centers_[:,1], marker='x', c="red")
    
    plt.show()
    #kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(emb)





#  _______  _______ _________          _______  ______     _______ 
# (       )(  ____ \\__   __/|\     /|(  ___  )(  __  \   / ___   )
# | () () || (    \/   ) (   | )   ( || (   ) || (  \  )  \/   )  |
# | || || || (__       | |   | (___) || |   | || |   ) |      /   )
# | |(_)| ||  __)      | |   |  ___  || |   | || |   | |    _/   / 
# | |   | || (         | |   | (   ) || |   | || |   ) |   /   _/  
# | )   ( || (____/\   | |   | )   ( || (___) || (__/  )  (   (__/\
# |/     \|(_______/   )_(   |/     \|(_______)(______/   \_______/

def method2(expls, max_num_elements_per_class, upper_bound=20):
    """
        Per class:
            - compute pairwise edit distances
            - recursive algorithm for prototype selection:
                - take graph most similar to others
                - do not consider all other graph which have a strong similarity with the ones added to the set
                - iterate until graphs are over or max number of graphs are added
    """
    for split in expls.keys():
        for c in expls[split].keys():
            class_expls = expls[split][c]

            m = np.full((len(class_expls), len(class_expls)), -10)
            for i in range(len(class_expls)):
                m[i, i] = 0
                for j in range(i+1, len(class_expls)):
                    tmp = nx.graph_edit_distance(class_expls[i][0], class_expls[j][0], upper_bound=upper_bound, timeout=60)
                    m[i, j] = tmp if tmp is not None else upper_bound
                    m[j, i] = m[i, j]

            # apply filtering algorithm
            prototypes , inertias , inertias2 = [] , [] , []
            print(f"-------------------------------------\nClass {c}")
            recursive_prototype_finding(m, prototypes, inertias, inertias2, max_num_elements=max_num_elements_per_class, remove=set())
            prototypes = prototypes[:max_num_elements_per_class] # can be replaced by elbow method

            ##
            # Plot metrics and results
            ##
            axs = plt.figure().subplots(1, 3)
            tmp = axs[0].matshow(m)
            #plt.colorbar(tmp, ax=axs[0])
            for i in range(len(m[0])):
                for j in range(len(m[0])):
                    axs[0].text(i, j, str(m[i,j]), va='center', ha='center')
            axs[0].set_title(f"Edit distances for Class {c}")

            # plot inertia curve            
            axs[1].plot(list(range(1, len(inertias)+1)), inertias)
            axs[1].set_title("Inertia")
            axs[1].set_xlabel("N° prototypes")
            axs[1].set_xticks(list(range(1, len(inertias)+1)))

            # plot inertia curve            
            axs[2].plot(list(range(1, len(inertias2)+1)), inertias2)
            axs[2].set_title("N° unmatched graphs")
            axs[2].set_xlabel("N° prototypes")
            axs[2].set_xticks(list(range(1, len(inertias2)+1)))
            axs[2].set_yticks(list(range(max(inertias2)+1)))

            # plot original vs filtered graphs
            fig = plt.figure(figsize=(7, 5), constrained_layout=True)
            fig.suptitle('Original vs Filtered Graphs')
            axs = fig.subplots(2, min(9, len(class_expls)))
            for j in range(min(9, len(class_expls))):
                plot_single_graph(class_expls[j][0], axs[0, j])
            for j , idx in enumerate(prototypes):         
                plot_single_graph(class_expls[idx][0], axs[1, j])            
            plt.show()



def inertia(proto, distances):
    """
        Compute the sum of the minimum Edit distances between prototypes and original graphs
        Only minimum prototype-original graphs edit distances are considered in the computation
    """
    ret = 0
    for i in range(len(distances[0])):
        min_index = np.argmin(distances[i, proto])
        #if sum(distances[proto[min_index]]) / (len(distances) - 1) <= 18: #if False, the 'i' graph is very similar to all other graphs, so consider as outlier
        ret += distances[i, proto[min_index]]
    return ret

def inertia2(proto, distances, M=7):
    """
        Compute the number of unmatched graphs in the original set w.r.t. the filtered set
        A graph is said unmatched if there are no graphs in the filtered set to which their Edit distance is <= M
    """
    ret = 0
    for i in range(len(distances[0])):
        min_distance = np.min(distances[i, proto])
        if min_distance > M:
            ret += 1
    return ret
    
def get_candidate_prototype(m, remove):
    """
        Return the idx of the graph to be selected, i.e., the one most similar to the majority of other graphs
    """
    avg_distances = []
    for i in range(len(m[0])):
        if i in remove:
            avg_distances.append(np.inf)
            continue

        i_avg_distance , n = 0 , 0
        for j in range(len(m[0])):
            if j not in remove:
                i_avg_distance += m[i, j]
                n += 1
        avg_distances.append(i_avg_distance / n)
    print("Avg distances: ", avg_distances)
    return np.argmin(avg_distances)

def remove_similar_to_prototype(m, prototypes, remove, threshold=4):
    """
        Add to 'remove' every graph which is too similar to the last added graph
    """
    remove.add(prototypes[-1])
    for i in range(len(m[0])):
        if m[prototypes[-1], i] <= threshold:
            remove.add(i)

def recursive_prototype_finding(m, prototypes, inertias, inertias2, max_num_elements, remove):
    """
        Greedy algorithm selecting at every iteration the element with minimum avg distance.
        Maybe a branch and bound works better?
    """
    if len(remove) == len(m[0]): #len(prototypes) >= max_num_elements or 
        return None
    else:
        idx = get_candidate_prototype(m, remove)
        print("Idx selected: ", idx)
        prototypes.append(idx)
        remove_similar_to_prototype(m, prototypes, remove)
        print("Remove set and inertia: ", remove, inertia(prototypes, m))
        inertias.append(inertia(prototypes, m))
        inertias2.append(inertia2(prototypes, m))
        return recursive_prototype_finding(m, prototypes, inertias, inertias2, max_num_elements, remove)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to use for explanations.')
    parser.add_argument('--dataset', default="", help='Dataset to explain.')
    parser.add_argument('--expl', default="", help='Explainer to use.')
    parser.add_argument('--time', default="", help='Time of reference experiment.')
    parser.add_argument('--k', default=5, help='Number of .', type=int)
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

    expls = read_explanations(path, divide_per_split=True, labels=fw.dataset.data.y, splits=["train"])
    print(expls.keys())

    #plot_edge_weight_distribution(expls, k=9) 

    expls = preprocess_explanations(expls, cut_edges=args.cut_edges, cut_cc=args.cut_cc)
    expls_unique = find_unique_explanations(expls, log=True)

    #plot_k_per_class(expls_unique, labels=fw.dataset.data.y, k=args.k)
    

    # METHOD 1: Extract features
    #embs = extract_features_per_graph(expls_unique, log=True)
    #visualize_embeddings(embs, k=args.k)


    # METHOD 2: Graph edit distance
    method2(expls_unique, max_num_elements_per_class=3)
    
