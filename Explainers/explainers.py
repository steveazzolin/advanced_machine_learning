from dig.xgraph.method import SubgraphX, PGExplainer
from dig.xgraph.method.subgraphx import find_closest_node_result

import networkx as nx
import torch
import joblib
from tqdm import tqdm
import numpy as np
from datetime import datetime
import os

from torch_geometric.utils import to_networkx, from_networkx, degree
from torch_geometric.data import Data


class Explainer():
    def __init__(self) -> None:
        pass

    def save_explanation(self, obj, path, name_file):
        os.makedirs(path, exist_ok=True)
        to_save = obj
        joblib.dump(to_save, path + name_file)

    def annotate_folder(self, path):
        params = self.get_params()
        with open(f'{path}/readme.txt', 'w') as f:
            for key , value in params.items():
                f.write(f'{key} = {value}\n')

    


class SemiSupSubGraphX(Explainer):
    def __init__(self, model, dataset, name_dataset, name_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = dataset.data
        self.model = model
        self.explainer = SubgraphX(model, num_classes=dataset.num_classes, device=self.device, explain_graph=False, high2low=True, reward_method='nc_mc_l_shapley')
        
        self.name_explainer = "SubGraphX"
        self.name_dataset = name_dataset
        self.name_model = name_model

    def explain(self, max_nodes, save=False):
        self.data.to(self.device)
        logits = self.model(self.data)

        explanations = []
        for node_idx in tqdm(range(self.data.num_nodes)):
            print("\nDegree: ", degree(self.data.edge_index[0], num_nodes=self.data.num_nodes)[node_idx].item())
            expl , pred = self.explain_node(node_idx, max_nodes, precomputed_logits=logits, visualize=False)
            explanations.append(expl)
            if save:
                partition = "None"
                if self.data.train_mask[node_idx]:
                    partition = "train"
                elif self.data.val_mask[node_idx]:
                    partition = "val"
                elif self.data.test_mask[node_idx]:
                    partition = "test"
                assert partition != "None"
                self.save_explanation(expl, f"Explanations/{self.name_explainer}/{self.name_dataset}/{self.name_model}/{partition}/{pred}/{node_idx}")
        return explanations
        

    def explain_node(self, node_idx, max_nodes, precomputed_logits=None, visualize=False):
        """
            Explain node 'node_idx'
            Returns a Data ogbject representing the explanation subgraph
        """
        if precomputed_logits is None:
            self.data.to(self.device)
            logits = self.model(self.data.x, self.data.edge_index)
        else:
            logits = precomputed_logits
        prediction = logits[node_idx].argmax(-1).item()

        _, explanation_results, related_preds = self.explainer(self.data.x, self.data.edge_index, node_idx=node_idx, max_nodes=max_nodes)
        explanation_results = explanation_results[prediction]
        explanation_results = self.explainer.read_from_MCTSInfo_list(explanation_results)
        result = find_closest_node_result(explanation_results, max_nodes=max_nodes)

        edge_list = [(n_frm, n_to) for (n_frm, n_to) in result.ori_graph.edges()
                        if n_frm in result.coalition and n_to in result.coalition]

        if visualize:            
            nx.draw_networkx(result.ori_graph, nodelist=result.coalition, edgelist=edge_list, pos=nx.spring_layout(result.ori_graph, seed=42))

        G = nx.Graph()
        for node in result.coalition:
            G.add_node(node)
        for edge in edge_list:
            G.add_edge(edge[0], edge[1])            
        data = from_networkx(G)
        return data , prediction









class SemiSupPGExplainer(Explainer):
    def __init__(self, framework, name_dataset, name_model, num_epochs=20, num_hops=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = framework.dataset
        self.model = framework.model
        self.framework = framework
        self.explainer = PGExplainer(self.model, in_channels=self.model.dim_embedding*3, device=self.device, explain_graph=False, epochs=num_epochs, num_hops=num_hops)
        
        self.name_explainer = "PGExplainer"
        self.name_dataset = name_dataset
        self.name_model = name_model
        self.num_epochs = num_epochs
        self.num_hops = num_hops

        self.gnn_acc , _ , _ = framework.predict(framework.train_loader, mask=framework.dataset.data.test_mask, return_metrics=True)
        print("Model's test accuracy: ", self.gnn_acc)

    def get_topk_edges_subgraph(self, edge_index, edge_mask, top_k, un_directed=False):
        if un_directed:
            top_k = 2 * top_k
        edge_mask = edge_mask.reshape(-1)
        thres_index = max(edge_mask.shape[0] - top_k, 0)
        threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
        hard_edge_mask = (edge_mask >= threshold)
        selected_edge_idx = np.where(hard_edge_mask == 1)[0].tolist()
        nodelist = []
        edgelist = []
        for edge_idx in selected_edge_idx:
            edges = edge_index[:, edge_idx].tolist()
            nodelist += [int(edges[0]), int(edges[1])]
            edgelist.append((edges[0], edges[1]))
        nodelist = list(set(nodelist))
        return nodelist, edgelist
    
    # def assemble_graph(self, nodelist, edgelist):
    #     G = nx.Graph()
    #     for node in nodelist:
    #         G.add_node(node)
    #     for edge in edgelist:
    #         G.add_edge(edge[0], edge[1])
    #     d = from_networkx(G)
    #     return d

        
    def explain_node(self, node_idx, top_k, precomputed_logits, precomputed_embd):
        with torch.no_grad():
            walks, masks, related_preds, edge_list = self.explainer(self.dataset.data.x, self.dataset.data.edge_index, node_idx=node_idx, y=self.dataset.data.y, top_k=top_k, logits=precomputed_logits, embed=precomputed_embd)
        edge_mask = masks[0].cpu()
        
        x, edge_index, y, subset, kwargs, mapping = self.explainer.get_subgraph(node_idx=node_idx, x=self.dataset.data.x, edge_index=self.dataset.data.edge_index, y=self.dataset.data.y)
        new_node_idx = torch.where(subset == node_idx)[0]
        # new_node_idx2 = mapping[node_idx]
        
        # new_data = Data(x=x, edge_index=edge_index)
        # graph = to_networkx(new_data)

        # edge_index = torch.tensor(list(graph.edges())).T
        # edge_mask = torch.FloatTensor(edge_mask)

        # nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k=top_k, un_directed=True)   
        # if new_node_idx not in nodelist:
        #     print("QUI")
        #     print(subset, new_node_idx)
        #     print(new_node_idx2)
        #     if new_node_idx2 in nodelist:
        #         print("La mia versio c'è")
        #     if new_node_idx in subset:
        #         print("Prima era qui")
        # ret = self.assemble_graph(nodelist, edgelist)
        # ret.target_node = int(new_node_idx)
                
        ret = nx.Graph(target_node=new_node_idx)
        for i , edge in enumerate(edge_list.cpu().T):
            edge = edge.tolist()
            ret.add_edge(edge[0], edge[1], weight=masks[0][i].item())
            ret.add_edge(edge[1], edge[0], weight=masks[0][i].item())
        return ret


    def explain(self, top_k, save=False):
        today = datetime.today().strftime('%Y-%m-%d-_%H-%M-%S')
        self.dataset.data.to(self.device)
        self.top_k = top_k
        
        print("Training PGExplainer...")
        self.pg_final_loss = self.explainer.train_explanation_network(self.dataset, precompute_netx=True, precompute_embds=False) 
        print("Training ended")

        preds , precomputed_logits = self.framework.predict(self.framework.train_loader, mask=torch.ones(self.dataset.data.num_nodes, dtype=torch.bool), return_logits=True)
        precomputed_embds = self.framework.get_embd(self.framework.train_loader)
        explanations = []
        for node_idx in tqdm(range(self.dataset.data.num_nodes)):
            expl = self.explain_node(node_idx, top_k=top_k, precomputed_logits=precomputed_logits, precomputed_embd=precomputed_embds)
            
            pred = preds[node_idx]
            explanations.append(expl)
            if save:
                partition = "None"
                if self.dataset.data.train_mask[node_idx]:
                    partition = "train"
                elif self.dataset.data.val_mask[node_idx]:
                    partition = "val"
                elif self.dataset.data.test_mask[node_idx]:
                    partition = "test"
                self.save_explanation(expl, f"Explanations/{self.name_explainer}/{self.name_dataset}/{self.name_model}/{today}/{partition}/{pred}/", f"{node_idx}.joblib")

        if save:
            self.annotate_folder(f"Explanations/{self.name_explainer}/{self.name_dataset}/{self.name_model}/{today}")
        return explanations
        
    def get_params(self):
        ret = {
            "num_epochs": self.num_epochs,
            "top_k": self.top_k,
            "num_hops": self.num_hops,
            "pg_final_loss": self.pg_final_loss,
            "gnn_test_accuracy": self.gnn_acc
        }
        return ret
    