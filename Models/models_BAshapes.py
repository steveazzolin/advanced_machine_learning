import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import BAShapes
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import argparse

try:
    from .framework import SemiSupFramework
    from .pyg_model_mask import GCNConvMask, GATConvMask
except ImportError:
    from framework import SemiSupFramework
    from pyg_model_mask import GCNConvMask, GATConvMask





def getFrameworkByName(model_name):
    if model_name == "GCN":
        ret = FrameworkBAShapes(model=GCN_BAShapes())
    elif model_name == "GAT":
        ret = FrameworkBAShapes(model=GAT_BAShapes())
    return ret


class FrameworkBAShapes(SemiSupFramework):
    def __init__(self, model, batch_size=1, main=False):        
        self.dataset = BAShapes()        
        optimizer = model.optimizer

        idx = torch.arange(self.dataset.data.num_nodes)
        train_idx , val_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=42)
        train_idx , test_idx = train_test_split(train_idx, train_size=0.8, stratify=self.dataset.data.y[train_idx], random_state=42)
        self.dataset.data.train_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.val_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.test_mask = torch.zeros(self.dataset.data.num_nodes, dtype=torch.bool)
        self.dataset.data.train_mask[train_idx] = True
        self.dataset.data.val_mask[val_idx] = True
        self.dataset.data.test_mask[test_idx] = True

        self.dataset.data.x[:, 0] = degree(self.dataset.data.edge_index[:,0], num_nodes=self.dataset.data.num_nodes) # otherways trainig fails with GAT, due to the features all having the same value
        self.dataset.data.x[:, 1] = self.dataset.data.x[:, 0]
        self.dataset.data.x[:, 2] = self.dataset.data.x[:, 0]
        self.dataset.data.x = self.dataset.data.x[:, :3]


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


class GCN_BAShapes(torch.nn.Module):
    def __init__(self, num_in_features=10, num_hidden=20, num_classes=4, dropout=0., lr=0.001, wd=0, num_epochs=1000):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.dim_embedding = num_hidden * 3

        self.convs = nn.ModuleList(
            [GCNConvMask(num_in_features, num_hidden)] + 
            [GCNConvMask(num_hidden, num_hidden) for _ in range(2)]
        )
        self.lin = nn.Linear(self.dim_embedding, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        input_lin = self.get_emb(data)
        out = self.lin(input_lin)
        return F.log_softmax(out, dim=1)

    def get_emb(self, data): 
        x, edge_index, edge_weights = data.x , data.edge_index ,  None
        stack = []

        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            stack.append(x)

        input_lin = torch.cat(stack, dim=1)
        return input_lin
    
    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU",
        }




class GAT_BAShapes(torch.nn.Module):
    def __init__(self, num_features=3, num_hidden=25, num_classes=4, num_heads=[1,1,1], dropout=0., lr=0.001, wd=0, num_epochs=700):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_epochs = num_epochs
        self.dim_embedding = num_hidden * num_heads[0] * num_heads[1] * 3
        
        self.convs = nn.ModuleList(
            [
                GATConvMask(num_features, num_hidden, heads=num_heads[0], dropout=dropout, concat=True),
                GATConvMask(num_hidden*num_heads[0], num_hidden*num_heads[0], heads=num_heads[1], dropout=dropout, concat=True),
                GATConvMask(num_hidden*num_heads[0]*num_heads[1], num_hidden*num_heads[0]*num_heads[1], heads=num_heads[2], dropout=dropout, concat=False)
            ]
        )
        self.lin = nn.Linear(self.dim_embedding, num_classes)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        x = self.get_emb(data)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "activation": "ELU",
        }

    def get_emb(self, data):
        x, edge_index, edge_weights = data.x , data.edge_index ,  None
        stack = []

        for conv in self.convs:
            x = conv(x, edge_index, edge_weights)
            x = F.elu(x)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            stack.append(x)

        input_lin = torch.cat(stack, dim=1)
        return input_lin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log training to wandb.')
    parser.add_argument('--train', action='store_true', default=False, help='To train the network from scratch.')
    parser.add_argument('--save', action='store_true', default=False, help='Whether to save the trained model or not.')
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.model == "GCN":
        gnn = GCN_BAShapes()
    elif args.model == "GAT":
        gnn = GAT_BAShapes()
    else:
        raise ValueError("Model unknown")

    fw = FrameworkBAShapes(model=gnn, main=True)
    if args.train:
        fw.train(log=True, log_wandb=args.wandb)
        if args.save:
            fw.save_model()
    else:
        print("Loading pretrained model...")
        fw.load_model()

    acc , preds , loss = fw.predict(fw.train_loader, fw.dataset.data.test_mask, return_metrics=True)
    print("Test accuracy: ", acc)
