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


class Framework:
    def __init__(self, model, train_loader, test_loader, optimizer, semi_sup, val_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.semi_sup = semi_sup
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader


    def train(self, num_epochs=200, log=False, log_wandb=False):
        if log_wandb:
            run = wandb.init(
                project='AML_project',
                name=self.model.__class__.__name__,
                entity='mcstewe',
                notes=str(self.model),
                reinit=True,
                save_code=True,
                config=dict (
                    epochs=num_epochs,
                    **self.model.get_hypers()
                ))
            wandb.watch(self.model)

        for epoch in range(1, num_epochs+1):
            loss = self.train_epoch(log)

            train_acc , _ , train_loss = self.predict(self.train_loader, predict_type="train", return_loss=True)
            if self.val_loader is not None:
                val_acc , _ , val_loss = self.predict(self.val_loader, predict_type="val", return_loss=True)
            else:
                val_acc , val_loss = 0 , 0
            test_acc , _ , test_loss = self.predict(self.test_loader, predict_type="test", return_loss=True)
            
            if epoch % 10 == 0 and log:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f} Val Acc: {val_acc:.3f}')
            if log_wandb:
                wandb.log({"train": {'loss': train_loss, "acc": train_acc}})
                wandb.log({"val": {'loss': val_loss, "acc": val_acc}})
                wandb.log({"test": {'loss': test_loss, "acc": test_acc}})
        
        if log_wandb:
            run.finish()
    
    def train_epoch(self, log):   
        self.model.train()
        self.optimizer.zero_grad()        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y) if not self.semi_sup else F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) # * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def predict(self, loader, predict_type, return_loss=False):
        assert predict_type in ("train", "val", "test")

        self.model.eval()
        n , total_correct ,  total_loss = 0 , 0 , 0
        preds = []
        for data in loader:
            if self.semi_sup:
                if predict_type == "val":
                    mask = data.val_mask
                if predict_type == "train":
                    mask = data.train_mask
                if predict_type == "test":
                    mask = data.test_mask

            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            batch_pred = out.argmax(-1).detach()
            
            if not self.semi_sup:
                total_correct += int((batch_pred == data.y).sum()) 
                preds.extend(batch_pred.cpu().tolist())
                if return_loss:
                    loss = F.nll_loss(out, data.y)
            else:
                total_correct += int((batch_pred[mask] == data.y[mask]).sum())
                preds.extend(batch_pred[mask].cpu().tolist())
                if return_loss:
                    loss = F.nll_loss(out[mask], data.y[mask])
            if return_loss:
                total_loss += float(loss.detach())
            
        acc = total_correct / len(preds)
        if return_loss:
            avg_loss = total_loss / len(loader)
            return acc , preds , avg_loss
        else:
            return acc , preds

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print("Model saved in: ", path)
        
    def load_model(self, path):        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()



class FrameworkCORA(Framework):
    def __init__(self, model, batch_size=1, lr=0.01, wd=5e-4):        
        self.dataset = Planetoid("../../Data/Cora","Cora")        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 

        train_loader = DataLoader(self.dataset)
        test_loader = DataLoader(self.dataset)
        val_loader = DataLoader(self.dataset)

        super().__init__(model, train_loader, test_loader, optimizer, semi_sup=True, val_loader=val_loader)



class GCN_CORA(torch.nn.Module):
    def __init__(self, num_in_features=1433, num_hidden=16, num_classes=7, dropout=0.5):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout

        self.gc1 = GCNConv(num_in_features, num_hidden)
        self.gc2 = GCNConv(num_hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None, **args):
        x = F.relu(self.gc1(x, edge_index))
        x = self.dropout(x)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_hypers(self):
        return {
            "num_hidden": self.num_hidden,
            "dropout": self.dropout
        }


class Framework_MUTAG(Framework):
    def __init__(self, model, batch_size=2048, lr=0.01, wd=1e-4):
        self.dataset = TUDataset("../../Data/mutagenicity","Mutagenicity")   
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 

        idx = torch.arange(len(self.dataset))
        train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y)

        train_loader = DataLoader(self.dataset[train_idx], batch_size=batch_size)
        test_loader = DataLoader(self.dataset[test_idx], batch_size=batch_size)

        super().__init__(model, train_loader, test_loader, optimizer, semi_sup=False)


class GCN_MUTAG(torch.nn.Module):
    def __init__(self, num_features=14, num_hidden=[64, 64, 32], num_classes=2, dropout=0.2):
        super().__init__()

        self.num_hidden = num_hidden
        self.dropout = dropout

        self.conv1 = GCNConv(num_features, num_hidden[0])
        self.conv2 = GCNConv(num_hidden[0], num_hidden[1])
        self.conv3 = GCNConv(num_hidden[1] , num_hidden[2])
        self.conv4 = GCNConv(num_hidden[2], num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for i in range(3)])

    def forward(self,x,edge_index,batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropouts[0](x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropouts[1](x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropouts[2](x)
        x = self.conv4(x, edge_index)
        x = global_max_pool(x, batch)
        return F.log_softmax(x, dim=-1)

    def get_hypers(self):
        return {
            "num_hidden": self.num_hidden,
            "dropout": self.dropout
        }
        

if __name__ == "__main__":
    #gcn = GCN_CORA()
    #fw = FrameworkCORA(model=gcn)
    #fw.train(log=True)

    gcn = GCN_MUTAG()
    fw = Framework_MUTAG(model=gcn)
    fw.train(log=True, log_wandb=True)
















class GCN_framework:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Planetoid("../../Data/Cora","Cora")
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GCNConv(num_features, 64)
                self.conv2 = GCNConv(64, 64)
                self.conv3 = GCNConv(64, 64)
                self.conv4 = GCNConv(64, num_classes)

            def forward(self,x,edge_index,batch):
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = self.conv4(x, edge_index)
                x = global_max_pool(x,batch)
                return F.log_softmax(x, dim=-1)
            

        self.model = Net(38,self.dataset.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01) 

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 201):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            


              
                
                

class GAT_framework:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Planetoid("../../Data/Cora","Cora")
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = GATConv(num_features, 64)
                self.conv2 = GATConv(64, 64)
                self.conv3 = GATConv(64, 64)
                self.conv4 = GATConv(64, num_classes)

            def forward(self,x,edge_index,batch):
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = self.conv4(x, edge_index)
                x = global_max_pool(x,batch)
                return F.log_softmax(x, dim=-1)
            

        self.model = Net(38,self.dataset.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01) 

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 201):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')
            



class GraphSAGE_framework:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = Planetoid("../../Data/Cora","Cora")
        class Net(torch.nn.Module):
            def __init__(self,num_features,num_classes):
                super().__init__()
                self.conv1 = SAGEConv(num_features, 64)
                self.conv2 = SAGEConv(64, 64)
                self.conv3 = GCNConv(64, 64)
                self.conv4 = GCNConv(64, num_classes)

            def forward(self,x,edge_index,batch):
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = self.conv4(x, edge_index)
                x = global_max_pool(x,batch)
                return F.log_softmax(x, dim=-1)
            

        self.model = Net(38,self.dataset.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.01) 

        idx = torch.arange(len(self.dataset))
        self.train_idx, self.test_idx = train_test_split(idx, train_size=0.8, stratify=self.dataset.data.y)

        self.train_loader = DataLoader(self.dataset[self.train_idx],batch_size=256)
        self.test_loader = DataLoader(self.dataset[self.test_idx],batch_size=256)
            
    def train(self):   
        self.model.train()
        self.optimizer.zero_grad()
        
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def test(self,loader):
        self.model.eval()

        total_correct = 0
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            total_correct += int((out.argmax(-1) == data.y).sum())
            
            loss = F.nll_loss(out, data.y)
            total_loss += float(loss) * data.num_graphs
            
        return total_correct / len(loader.dataset),total_loss / len(self.train_loader.dataset)

    def iterate(self):

        for epoch in range(1, 201):
            loss = self.train()
            train_acc,train_loss = self.test(self.train_loader)
            test_acc,test_loss = self.test(self.test_loader)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.3f}, Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} '
                f'Test Acc: {test_acc:.3f}')


    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print("model saved in: ",path)
        
    def load_model(self,path):
        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
    def evaluate(self):

        train_acc,train_loss = self.test(self.train_loader)
        test_acc,test_loss = self.test(self.test_loader)
        print(f'Test Loss: {test_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}')