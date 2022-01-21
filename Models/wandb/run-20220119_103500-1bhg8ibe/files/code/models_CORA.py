from torch_geometric.nn import GCN

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv,global_max_pool, GATConv,SAGEConv
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import argparse

from framework import Framework



class FrameworkCORA(Framework):
    def __init__(self, model, batch_size=1):        
        self.dataset = Planetoid("../../Data/Cora","Cora")        
        optimizer = model.optimizer

        train_loader = DataLoader(self.dataset)
        test_loader = DataLoader(self.dataset)
        val_loader = DataLoader(self.dataset)

        super().__init__(model, train_loader, test_loader, optimizer, semi_sup=True, val_loader=val_loader)


class GCN_CORA(torch.nn.Module):
    def __init__(self, num_in_features=1433, num_hidden=16, num_classes=7, dropout=0.5, lr=0.01, wd=5e-4):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout

        self.gc1 = GCNConv(num_in_features, num_hidden)
        self.gc2 = GCNConv(num_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.gc1(x, edge_index))
        x = self.dropout(x)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_hypers(self):
        return {
            "num_hidden": self.num_hidden,
            "dropout": self.dropout
        }

        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="",
                        help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, 
                        help='Log training to wandb.')
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.model == "GCN":
        gnn = GCN_CORA()
    elif args.model == "GAT":
        pass
        #gnn = GAT_CORA()
    else:
        raise ValueError("Model unknown")

    fw = FrameworkCORA(model=gnn)
    fw.train(log=True, log_wandb=args.wandb)















