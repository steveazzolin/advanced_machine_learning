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

from framework import SemiSupFramework



class FrameworkCORA(SemiSupFramework):
    def __init__(self, model, batch_size=1):        
        self.dataset = Planetoid("../../Data/Cora","Cora")        
        optimizer = model.optimizer

        train_loader = DataLoader(self.dataset, batch_size=batch_size)
        test_loader = DataLoader(self.dataset, batch_size=batch_size)
        val_loader = DataLoader(self.dataset, batch_size=batch_size)

        super().__init__(model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=optimizer, 
                        num_epochs=model.num_epochs, 
                        semi_sup=True, 
                        val_loader=val_loader)


class GCN_CORA(torch.nn.Module):
    def __init__(self, num_in_features=1433, num_hidden=16, num_classes=7, dropout=0.5, lr=0.01, wd=5e-4, num_epochs=200):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
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
            "dropout": self.dropout,
            "activation": "ReLU",
            "weight_decay": self.optimizer.param_groups[0]["lr"],
            "learning_rate": self.optimizer.param_groups[0]["weight_decay"],
            "optimizer": self.optimizer.__class__.__name__,
            "num_epochs": self.num_epochs
        }



class GAT_CORA(torch.nn.Module):
    """
    Hyper-parameters from https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L8
    """
    def __init__(self, num_features=1433, num_hidden=8, num_classes=7, num_heads=[8,1], dropout=0.6, lr=5e-3, wd=5e-4, num_epochs=500):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_epochs = num_epochs

        self.conv1 = GATConv(num_features, num_hidden, heads=num_heads[0], dropout=dropout, concat=False)
        self.conv2 = GATConv(num_hidden, num_classes, heads=num_heads[1], dropout=dropout, concat=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "num_hidden": self.num_hidden,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "activation": "ELU",
            "weight_decay": self.optimizer.param_groups[0]["lr"],
            "learning_rate": self.optimizer.param_groups[0]["weight_decay"],
            "optimizer": self.optimizer.__class__.__name__,
            "num_epochs": self.num_epochs
        }

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log training to wandb.')
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.model == "GCN":
        gnn = GCN_CORA()
    elif args.model == "GAT":
        gnn = GAT_CORA()
    else:
        raise ValueError("Model unknown")

    fw = FrameworkCORA(model=gnn)
    fw.train(log=True, log_wandb=args.wandb)















