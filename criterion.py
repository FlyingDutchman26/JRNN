import torch
import torch.nn as nn

class JRNNLoss(nn.Module):
    def __init__(self, weight_trg = None, weight_arg = None):
        super().__init__()
        self.loss_trg = nn.CrossEntropyLoss(weight_trg)
        self.loss_arg = nn.CrossEntropyLoss(weight_arg)

        
    def forward(self,T,T_label,A_pred,A_label,entity_idx):
        # print('debug: T', T.shape)
        # print('debug: T_labe;',T_label.shape )
        T = T.reshape(-1,35)
        T_label = T_label.reshape(-1)
        loss_1 = self.loss_trg(T,T_label)
        if len(entity_idx) == 0:
            loss_2 = 0
        else:
            A_label = A_label.reshape([-1,A_label.shape[1]]) # [B,L,L] -> [B*L,L]
            A_label = A_label[entity_idx,:]
            A_pred = A_pred.reshape(-1,24)
            A_label = A_label.reshape(-1)
            loss_2 = self.loss_arg(A_pred,A_label)
            loss = loss_1 + loss_2
        return loss,loss_1,loss_2