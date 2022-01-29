import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GCNConv, global_max_pool, GATConv, SAGEConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import argparse

try:
    from .framework import GraphClassificationFramework
except ImportError:
    from framework import GraphClassificationFramework


def getFrameworkByName(model_name):
    if model_name == "GCN":
        ret = Framework_MUTAG(model=GCN_MUTAG())
    elif model_name == "GAT":
        ret = Framework_MUTAG(model=GAT_MUTAG())
    return ret


class Framework_MUTAG(GraphClassificationFramework):
    def __init__(self, model, batch_size):
        self.dataset = TUDataset("../../Data/mutagenicity","Mutagenicity")   

        idx = torch.arange(len(self.dataset))
        train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y, random_state=42)

        train_loader = DataLoader(self.dataset[train_idx], batch_size=batch_size)
        test_loader = DataLoader(self.dataset[test_idx], batch_size=batch_size)

        super().__init__(model=model, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        optimizer=model.optimizer, 
                        num_epochs=model.num_epochs, 
                        semi_sup=False)




class GCN_MUTAG(torch.nn.Module):
    def __init__(self, num_features=14, num_hidden=[64, 64, 32], num_classes=2, dropout=0.2, lr=0.01, wd=1e-4, num_epochs=200):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.dropout = dropout

        self.conv1 = GCNConv(num_features, num_hidden[0])
        self.conv2 = GCNConv(num_hidden[0], num_hidden[1])
        self.conv3 = GCNConv(num_hidden[1] , num_hidden[2])
        self.conv4 = GCNConv(num_hidden[2], num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(3)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def forward(self, data):
        x, edge_index, batch = data.x , data.edge_index , data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropouts[0](x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropouts[1](x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropouts[2](x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "num_hidden": self.num_hidden,
            "dropout": self.dropout,
            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer": self.optimizer.__class__.__name__,
            "num_epochs": self.num_epochs
        }



class GAT_MUTAG(torch.nn.Module):
    """
        To fix: On Windows it does not learn
    """
    def __init__(self, num_features=14, num_hidden=[64, 64, 64], num_classes=2, dropout=0.2, lr=0.01, wd=1e-4, num_epochs=200):
        super().__init__()

        self.num_epochs = num_epochs
        self.dropout = dropout
        self.num_hidden = num_hidden

        self.conv1 = GATConv(num_features, 64)
        self.conv2 = GATConv(64, 64)
        self.conv3 = GATConv(64 , 64)
        self.conv4 = GATConv(64, num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(4)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        x, edge_index, batch = data.x , data.edge_index , data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropouts[0](x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropouts[1](x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropouts[2](x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.dropouts[3](x)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU"
        }



class GraphSAGE_MUTAG(torch.nn.Module):
    def __init__(self, num_features=14, num_hidden=[64, 64, 64], num_classes=2, dropout=0.2, lr=0.01, wd=1e-4, num_epochs=200):
        super().__init__()

        self.num_epochs = num_epochs
        self.dropout = dropout
        self.num_hidden = num_hidden

        self.conv1 = SAGEConv(num_features, num_hidden[0])
        self.conv2 = SAGEConv(num_hidden[0], num_hidden[1])
        self.conv3 = SAGEConv(num_hidden[1], num_hidden[2])
        self.conv4 = SAGEConv(num_hidden[2], num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(3)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def forward(self, data):
        x, edge_index, batch = data.x , data.edge_index , data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropouts[0](x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropouts[1](x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropouts[2](x)
        x = self.conv4(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU"
        }
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log training to wandb.')
    parser.add_argument('--train', action='store_true', default=False, help='Train the network from scratch.')
    parser.add_argument('--save', action='store_true', default=False, help='Save the trained network.')
    parser.add_argument('--batch_size', default=256, help='Batch size.', type=int)
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.model == "GCN":
        gnn = GCN_MUTAG()
    elif args.model == "SAGE":
        gnn = GraphSAGE_MUTAG(num_epochs=500)
    elif args.model == "GAT":
        gnn = GAT_MUTAG()
    else:
        raise ValueError("Model unknown")

    fw = Framework_MUTAG(model=gnn, batch_size=args.batch_size)
    if args.train:
        fw.train(log=True, log_wandb=args.wandb)
        if args.save:
            fw.save_model()
    else:
        print("Loading pretrained model...")
        fw.load_model()

    acc , _ = fw.predict(fw.test_loader)
    print("Test accuracy: ", acc)




