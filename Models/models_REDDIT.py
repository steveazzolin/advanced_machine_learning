import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.datasets import Reddit2, Reddit, Planetoid
from torch_geometric.loader import DataLoader, NeighborLoader
import argparse
import copy

try:
    from .framework import LargeSemiSupFramework
    from . import utils
except ImportError:
    from framework import LargeSemiSupFramework
    import utils




def getFrameworkByName(model_name, batch_size, num_workers):
    if model_name == "GCN":
        ret = FrameworkREDDIT(model=GCN_REDDIT(), batch_size=batch_size, num_workers=num_workers)
    if model_name == "SAGE":
        ret = FrameworkREDDIT(model=SAGE_REDDIT(), batch_size=batch_size, num_workers=num_workers)
    elif model_name == "GAT":
        ret = FrameworkREDDIT(model=GAT_REDDIT(), batch_size=batch_size, num_workers=num_workers)
    return ret

class FrameworkREDDIT(LargeSemiSupFramework):
    """
        Code partially from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
    """
    def __init__(self, model, batch_size, num_workers, persistent_workers=True, main=False):        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Reddit(self.get_data_path() + "Reddit")  #Planetoid(root=r'../../Data/Cora', name='Cora') 
        optimizer = model.optimizer
        self.data = self.dataset[0].to(device, 'x', 'y')

        g = torch.Generator()  #for reproducibility
        g.manual_seed(42)

        kwargs = {'batch_size': batch_size, 'num_workers': num_workers,  'persistent_workers': persistent_workers, "pin_memory": True, "worker_init_fn": utils.seed_worker, "generator": g}
        train_loader = NeighborLoader(self.data, input_nodes=self.dataset.data.train_mask, num_neighbors=[25, 10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(self.data), input_nodes=None, num_neighbors=[-1], shuffle=False, **kwargs)
        print("Dataset loaded")
        
        del subgraph_loader.data.x , subgraph_loader.data.y
        subgraph_loader.data.num_nodes = self.dataset.data.num_nodes
        subgraph_loader.data.n_id = torch.arange(self.dataset.data.num_nodes)
        train_loader.data.n_id = torch.arange(self.dataset.data.num_nodes)

        super().__init__(model=model, 
                        train_loader=train_loader, 
                        test_loader=None, 
                        optimizer=optimizer, 
                        num_epochs=model.num_epochs, 
                        semi_sup=True, 
                        val_loader=None,
                        subgraph_loader=subgraph_loader,
                        main=main)




class GCN_REDDIT(torch.nn.Module):
    def __init__(self, num_in_features=602, num_hidden=256, num_classes=42, dropout=0.5, lr=0.01, wd=0, num_epochs=10):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.dim_embedding = num_classes

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_in_features, num_hidden))
        self.convs.append(GCNConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def get_emb(self, data, device):
        x , edge_index = data.x, data.edge_index.to(device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = self.dropout(x)
        return x

    def forward(self, data, device):
        x = self.get_emb(data, device)        
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        """
            Compute representations of nodes layer by layer, using *all* available edges. 
            This leads to faster computation in contrast to immediately computing the final representations of each batch

            In case of Out-of-Memory issues, check the original script for moving 'xs' to the cpu into the for loop
        """
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device) #take features of [nodes, their_neighborhood] in the subgraph batch
                x = conv(x, batch.edge_index.to(device))  # apply conv
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu()) #append just the rerpsentations of target nodes (the first batch_size elem)
            x_all = torch.cat(xs, dim=0) # the new matrix of node representations
        x_all = x_all.to(device)
        return F.log_softmax(x_all, dim=1)
    
    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU",
        }



class SAGE_REDDIT(torch.nn.Module):
    def __init__(self, num_in_features=602, num_hidden=256, num_classes=42, dropout=0.5, lr=0.01, wd=0, num_epochs=10):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.dim_embedding = num_classes

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_in_features, num_hidden))
        self.convs.append(SAGEConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def get_emb(self, data):
        x , edge_index = data.x, data.edge_index.to(self.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = self.dropout(x)
        return x

    def forward(self, data):
        x = self.get_emb(data)        
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        """
            Compute representations of nodes layer by layer, using *all* available edges. 
            This leads to faster computation in contrast to immediately computing the final representations of each batch
        """
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device) #take features of [nodes, their_neighborhood] in the subgraph batch
                x = conv(x, batch.edge_index.to(device))  # apply conv
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu()) #append just the rerpsentations of target nodes (the first batch_size elem)
            x_all = torch.cat(xs, dim=0) # the new matrix of node representations
        x_all = x_all.to(device)
        return F.log_softmax(x_all, dim=1)
    
    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "activation": "ReLU",
        }



class GAT_REDDIT(torch.nn.Module):
    """
    Hyper-parameters from https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L8
    """
    def __init__(self, num_features=602, num_hidden=256, num_classes=42, num_heads=[8,1], dropout=0.6, lr=5e-3, wd=0, num_epochs=20):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_epochs = num_epochs
        self.dim_embedding = num_classes

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, num_hidden, heads=num_heads[0], dropout=dropout, concat=True))
        self.convs.append(GATConv(num_hidden*num_heads[0], num_classes, heads=num_heads[1], dropout=dropout, concat=False))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd) 

    def get_emb(self, data):
        x , edge_index = data.x, data.edge_index.to(self.device)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = self.dropout(x)
        return x

    def forward(self, data):
        x = self.get_emb(data)        
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        """
            Compute representations of nodes layer by layer, using *all* available edges. 
            This leads to faster computation in contrast to immediately computing the final representations of each batch
        """
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device) #take features of [nodes, their_neighborhood] in the subgraph batch
                x = conv(x, batch.edge_index.to(device))  # apply conv
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu()) #append just the rerpsentations of target nodes (the first batch_size elem)
            x_all = torch.cat(xs, dim=0) # the new matrix of node representations
        x_all = x_all.to(device)
        return F.log_softmax(x_all, dim=1)

    def get_hypers(self):
        return {
            "dropout": self.dropout,
            "num_heads": self.num_heads,
            "activation": "ReLU",
        }

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="", help='Model to train.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Log training to wandb.')
    parser.add_argument('--train', action='store_true', default=False, help='To train the network from scratch.')
    parser.add_argument('--save', action='store_true', default=False, help='Whether to save the trained model or not.')
    parser.add_argument('--batch_size', default=512, help='Batch size.', type=int)  
    parser.add_argument('--num_workers', default=6, help='Num workers.')
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.model == "GCN":
        gnn = GCN_REDDIT(num_epochs=10)
        bs = min(2048*2, args.batch_size)
    elif args.model == "SAGE":
        gnn = SAGE_REDDIT(num_epochs=10)
        bs = min(2048*3, args.batch_size)
    elif args.model == "GAT":
        gnn = GAT_REDDIT(num_epochs=20)
        bs = min(2048, args.batch_size)
    else:
        raise ValueError("Model unknown")

    fw = FrameworkREDDIT(model=gnn, batch_size=bs, num_workers=args.num_workers, main=True)
    if args.train:
        fw.train(log=True, log_wandb=args.wandb)
        if args.save:
            fw.save_model()
    else:
        print("Loading pretrained model...")
        fw.load_model()

    del fw.train_loader, fw.subgraph_loader
    kwargs = {'batch_size': 256, 'num_workers': 0, "pin_memory": True}
    tmp = NeighborLoader(fw.dataset.data, input_nodes=None, num_neighbors=[-1], **kwargs)
    tmp.data.n_id = torch.arange(tmp.data.num_nodes)
    acc , _ = fw.predict(tmp, mask=tmp.data.test_mask, return_metrics=True)
    print("Test accuracy: ", acc)