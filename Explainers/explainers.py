from dig.xgraph.method import SubgraphX
from dig.xgraph.method.subgraphx import find_closest_node_result

import networkx as nx
import torch
import joblib
from tqdm import tqdm

from torch_geometric.utils import to_networkx, from_networkx, degree


class Explainer():
    def __init__(self) -> None:
        pass

    def save_explanation(self, obj, path):
        to_save = obj.to_dict()
        joblib.dump(to_save, path)


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
                if self.data.test_mask[node_idx]:
                    partition = "test"
                self.save_explanation(expl, f"../{self.name_explainer}/{self.name_dataset}/{self.name_model}/{partition}/{pred}/{node_idx}")
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