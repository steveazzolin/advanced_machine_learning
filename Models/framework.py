from sys import prefix
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import wandb
import copy
import time
import os

from utils import EarlyStopping


class Framework:
    def __init__(self, model, train_loader, test_loader, optimizer, num_epochs, semi_sup, val_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.semi_sup = semi_sup
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.path = "pretrained/"

    def save_model(self, prefix=""):
        p = self.path + prefix + self.model.__class__.__name__ + ".pt"
        torch.save(self.model.state_dict(), p)
        
    def load_model(self, prefix=""):        
        self.model.load_state_dict(torch.load(self.path + prefix + self.model.__class__.__name__ + ".pt"))
        self.model.eval()

    def delete_model(self, prefix=""):
        assert prefix != "" , "Can delete just temporary models"
        p = self.path + prefix + self.model.__class__.__name__ + ".pt"
        if os.path.exists(p):
            os.remove(p)
        else:
            print("You are trying to delete a model that does not exists")

    def init_logger(self):
        """
            Init wandb settings
        """
        run = wandb.init(
                project='AML_project',
                name=self.model.__class__.__name__,
                entity='mcstewe',
                notes=str(self.model),
                reinit=True,
                save_code=True,
                config=dict(
                    {
                        "num_hidden": self.model.num_hidden,
                        "weight_decay": self.model.optimizer.param_groups[0]["weight_decay"],
                        "learning_rate": self.model.optimizer.param_groups[0]["lr"],
                        "optimizer": self.model.optimizer.__class__.__name__,
                        "num_epochs": self.model.num_epochs,
                    }
                    **self.model.get_hypers()
                ))
        wandb.watch(self.model)
        return run

    def stop_logger(self, run):
        run.finish()

    def log(self, msg, to_wandb=False):
        if not to_wandb:
            print(msg)
        else:
            wandb.log(msg)




class SemiSupFramework(Framework):
    """
        Default framework for semi-supervised node classification (e.g. Cora)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.train_loader.dataset.data.val_mask.sum() > 0 , "This class assumes a validation split"

    def train(self, log=False, log_wandb=False):
        if log_wandb:
            run = self.init_logger()

        best_val_loss = np.inf
        for epoch in range(1, self.num_epochs+1):
            train_loss = self.train_epoch()
            train_acc , val_acc , val_loss , test_acc = self.predict(self.train_loader, is_training=True, return_metrics=True)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                self.save_model(prefix="best_so_far_")
            
            if epoch % 10 == 0 and log:
                self.log(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f} '
                         f'Test Acc: {test_acc:.3f} Val Acc: {val_acc:.3f}')
            if log_wandb:
                self.log({"train": {'loss': train_loss, "acc": train_acc}}, to_wandb=True)
                self.log({"val": {'loss': val_loss, "acc": val_acc}}, to_wandb=True)
                self.log({"test": {"acc": test_acc}}, to_wandb=True)
        
        if log_wandb:
            self.stop_logger(run)
        self.load_model(prefix="best_so_far_")
        #self.delete_model(prefix="best_so_far_")
    
    def train_epoch(self):
        assert len(self.train_loader) == 1

        self.model.train()    
        train_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index)
            train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            self.optimizer.step()      
        return train_loss.item()

    @torch.no_grad()
    def predict(self, loader, mask=None, is_training=False, return_metrics=False):
        assert len(self.train_loader) == 1

        self.model.eval()
        total_correct = 0 
        preds = []
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            batch_pred = out.argmax(-1).detach()
            preds.extend(batch_pred[mask].cpu().tolist())

            if return_metrics:
                if is_training:
                    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]).item()
                    train_acc = float((batch_pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()) 
                    val_acc = float((batch_pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()) 
                    test_acc = float((batch_pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum())   
                else:
                    total_correct += int((batch_pred[mask] == data.y[mask]).sum())                
                    loss = F.nll_loss(out[mask], data.y[mask])
            
        if return_metrics:
            if is_training:
                return train_acc, val_acc , val_loss , test_acc
            else:
                acc = total_correct / len(preds)
                return acc , preds , loss
        else:
            return preds


class LargeSemiSupFramework(Framework):
    """
        Framework optimized for large scale semi-supervised node classification (e.g. REDDIT)
    """
    def __init__(self, subgraph_loader, **kwargs):
        super().__init__(**kwargs)
        self.subgraph_loader = subgraph_loader

    def train(self, log=False, log_wandb=False):
        if log_wandb:
            run = self.init_logger()

        best_val_loss = np.inf
        for epoch in range(1, self.num_epochs+1):
            start_time = time.time()
            train_loss = self.train_epoch()
            end_time = time.time()
            train_acc , val_acc , val_loss , test_acc = self.test(self.train_loader, self.subgraph_loader)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss    
                self.save_model(prefix="best_so_far_")        
            if epoch % 1 == 0 and log:
                self.log(f'Epoch: {epoch:03d}, Time: {end_time - start_time:.1f} Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f} Val Acc: {val_acc:.3f}')
            if log_wandb:
                self.log({"train": {'loss': train_loss, "acc": train_acc}}, to_wandb=True)
                self.log({"val": {'loss': val_loss, "acc": val_acc}}, to_wandb=True)
                self.log({"test": {"acc": test_acc}}, to_wandb=True)
        
        if log_wandb:
            self.stop_logger(run)
        self.load_model(prefix="best_so_far_")
    
    def train_epoch(self):
        self.model.train()    
        total_loss = 0 
        for data in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index.to(self.device))[:data.batch_size]
            train_loss = F.nll_loss(output, data.y[:data.batch_size])
            train_loss.backward()
            self.optimizer.step()
            total_loss += train_loss.detach()      
        return float(total_loss.item() / len(self.train_loader))

    @torch.no_grad()
    def test(self, train_loader, subgraph_loader):
        self.model.eval()
        out = self.model.inference(train_loader.data.x, subgraph_loader, self.device)
        batch_pred = out.argmax(-1).detach()

        val_loss = F.nll_loss(out[train_loader.data.val_mask], train_loader.data.y[train_loader.data.val_mask]).item()
        train_acc = (batch_pred[train_loader.data.train_mask] == train_loader.data.y[train_loader.data.train_mask]).sum() / train_loader.data.train_mask.sum()
        val_acc = (batch_pred[train_loader.data.val_mask] == train_loader.data.y[train_loader.data.val_mask]).sum() / train_loader.data.val_mask.sum()
        test_acc = (batch_pred[train_loader.data.test_mask] == train_loader.data.y[train_loader.data.test_mask]).sum() / train_loader.data.test_mask.sum()

        return train_acc , val_acc , val_loss , test_acc

    @torch.no_grad()
    def predict(self, loader, mask=None, return_metrics=False):
        assert loader.data.n_id is not None

        self.model.eval()
        out = self.model.inference(loader.data.x, loader, self.device)
        preds = out.argmax(-1).detach()

        acc = float((preds[mask] == loader.data.y[mask]).sum() / mask.sum())
        if return_metrics:
            return acc , preds
        else:
            return preds




class GraphClassificationFramework(Framework):
    """
        Default framework for graph classification tasks
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, log=False, log_wandb=False):
        if log_wandb:
            run = self.init_logger()

        best_val_loss = 0
        val_acc , val_loss = 0 , 0
        for epoch in range(1, self.num_epochs+1):
            train_loss = self.train_epoch(log)

            train_acc , _  = self.predict(self.train_loader, predict_type="train", return_loss=False)
            test_acc  , _  = self.predict(self.test_loader, predict_type="test", return_loss=False)
            if self.val_loader is not None:
                val_acc , _ , val_loss = self.predict(self.val_loader, predict_type="val", return_loss=True)
                if val_loss <= best_val_loss:
                    self.save_model(prefix="best_so_far_")
                    best_val_loss = val_loss

            if epoch % 10 == 0 and log:
                self.log(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Train Acc: {train_acc:.3f} '
                         f'Test Acc: {test_acc:.3f} Val Acc: {val_acc:.3f}')
            if log_wandb:
                self.log({"train": {'loss': train_loss, "acc": train_acc}}, to_wandb=True)
                self.log({"val": {'loss': val_loss, "acc": val_acc}}, to_wandb=True)
                self.log({"test": {"acc": test_acc}}, to_wandb=True)        
        if log_wandb:
            self.stop_logger(run)
        if self.val_loader is not None:
            self.load_model(prefix="best_so_far_")
    
    def train_epoch(self, log):   
        self.model.train()    
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach()
        return float(total_loss / len(self.train_loader))

    @torch.no_grad()
    def predict(self, loader, predict_type="test", return_loss=False):
        assert predict_type in ("train", "val", "test")

        self.model.eval()
        total_correct ,  total_loss = 0 , 0
        preds = []
        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            
            batch_pred = out.argmax(-1).detach()                        
            total_correct += int((batch_pred == data.y).sum()) 
            preds.extend(batch_pred.cpu().tolist())
            if return_loss:
                loss = F.nll_loss(out, data.y)
                total_loss += float(loss.detach())            
        acc = total_correct / len(preds)
        if return_loss:
            avg_loss = total_loss / len(loader)
            return acc , preds , avg_loss
        else:
            return acc , preds