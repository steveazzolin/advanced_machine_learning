import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import argparse

try:
    from .framework import SemiSupFramework
    from .pyg_model_mask import GCNConvMask, GATConvMask
except ImportError:
    from framework import SemiSupFramework
    from pyg_model_mask import GCNConvMask, GATConvMask





def getFrameworkByName(model_name):
    if model_name == "GCN":
        ret = FrameworkCITESEER(model=GCN_CITESEER())
    elif model_name == "GAT":
        ret = FrameworkCITESEER(model=GAT_CITESEER())
    return ret


class FrameworkCITESEER(SemiSupFramework):
    def __init__(self, model, batch_size=1, main=False):        
        self.dataset = Planetoid(self.get_data_path() + "CITESEER","CITESEER")        
        optimizer = model.optimizer

        train_loader = DataLoader(self.dataset, batch_size=batch_size)
        test_loader = None
        val_loader = None

        super().__init__(model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=optimizer, 
                        num_epochs=model.num_epochs, 
                        semi_sup=True, 
                        val_loader=val_loader,
                        main=main)


class GCN_CITESEER(torch.nn.Module):
    def __init__(self, num_in_features=3703, num_hidden=100, num_classes=6, dropout=0.7, lr=0.0001, wd=11e-3, num_epochs=500):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.dim_embedding = num_hidden

        self.gc1 = GCNConvMask(num_in_features, num_hidden)
        self.gc2 = GCNConvMask(num_hidden, num_hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(num_hidden, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        x = self.get_emb(data)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

    def get_emb(self, data):
        x, edge_index = data.x , data.edge_index
        x = F.relu(self.gc1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, edge_index))
        x = self.dropout(x)
        return x
    
    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU",
        }




class GAT_CITESEER(torch.nn.Module):
    """
    Hyper-parameters from https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L8
    """
    def __init__(self, num_features=3703, num_hidden=100, num_classes=6, num_heads=[2,1], dropout=0.7, lr=0.0001, wd=5e-4, num_epochs=200):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_epochs = num_epochs
        self.dim_embedding = num_hidden*num_heads[0]

        self.conv1 = GATConvMask(num_features, num_hidden, heads=num_heads[0], dropout=dropout, concat=True)
        self.conv2 = GATConvMask(num_hidden*num_heads[0], num_hidden*num_heads[0], heads=num_heads[1], dropout=dropout, concat=False)
        self.lin = nn.Linear(num_hidden*num_heads[0], num_classes)
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        x = self.get_emb(data)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "GAT_dropout": self.dropout,
            "num_heads": self.num_heads,
            "activation": "ELU",
        }

    def get_emb(self, data):
        x, edge_index = data.x , data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        return x

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log training to wandb.')
    parser.add_argument('--train', action='store_true', default=False, help='To train the network from scratch.')
    parser.add_argument('--save', action='store_true', default=False, help='Whether to save the trained model or not.')
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.model == "GCN":
        gnn = GCN_CITESEER()
    elif args.model == "GAT":
        gnn = GAT_CITESEER()
    else:
        raise ValueError("Model unknown")

    fw = FrameworkCITESEER(model=gnn, main=True)
    if args.train:
        fw.train(log=True, log_wandb=args.wandb)
        if args.save:
            fw.save_model()
    else:
        print("Loading pretrained model...")
        fw.load_model()

    acc , preds , loss = fw.predict(fw.train_loader, fw.dataset.data.test_mask, return_metrics=True)
    print("Test accuracy: ", acc)
