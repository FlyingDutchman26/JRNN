import torch
import torch.nn as nn


class Runner:
    def __init__(self,model,optimizer,loss_function) -> None:
        self.model =model
        self.optimizer = optimizer
        self.loss_fn = loss_function
        self.loss_fn = self.loss_fn.double()
    
    def train(self,train_loader,num_epoch=1):
        step = 0
        for epoch in range(1,num_epoch+1):
            for batch_id, (X, E, T_label, A_label, valid_len) in enumerate(train_loader):
                step += 1
                self.model.train()
                T, T_pred, pred_trg_idx, A_pred, entity_idx = self.model(X,E)
                
                loss ,loss_1, loss_2= self.loss_fn(T,T_label,A_pred,A_label,entity_idx)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    t = T.reshape(-1,35)
                    t = torch.argmax(t,dim=1)
                    t_label = T_label.reshape(-1)
                    score = (t == t_label).sum()/len(t_label)
                if step ==1 or step % 50 == 0:
                    print(f'[epoch]:{epoch},[step]:{step},[loss]:{loss.item(),loss_1.item(),loss_2.item()},[score]:{score}')
                
                
                
