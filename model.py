import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_weight, entity_type_dim):
        super().__init__()
        vocab_size = embedding_weight.shape[0] 
        word_ebd_size = embedding_weight.shape[1] # 100
        self.word_embedding = nn.Embedding(vocab_size,word_ebd_size,padding_idx=1)
        self.word_embedding.weight = nn.Parameter(embedding_weight)
        self.entity_type_embedding = nn.Embedding(9,entity_type_dim,padding_idx=1) # 7种实体，1个<pad>,一个<unk>代表不是实体
    
    def forward(self, X, E):
        '''
        Input X,E: sentence vector:[B,L], entity vector: [B,L] 
        
        output: W: [B,L,d_x + d_e]
        
        '''
        X = self.word_embedding(X)
        E = self.entity_type_embedding(E)
        W = torch.cat([X,E],dim=2)
        W = W.double()
        return W

class EncodingLayer(nn.Module):
    def __init__(self, embedding_weight, entity_type_dim, hidden_size, num_layers, dropout = 0, bidirectional = True, **kwargs) -> None:
        super().__init__()
        input_size = embedding_weight.shape[1] + entity_type_dim
        self.embedding = EmbeddingLayer(embedding_weight, entity_type_dim)
        self.rnn = nn.GRU(input_size,hidden_size,num_layers,dropout = dropout,bidirectional = bidirectional)
        self.rnn = self.rnn.double()
    def forward(self, X, E):
        '''
        Input: X,E: [B,L]
        
        Output: H : [L,B,d], d = d_w + d_e
        '''
        W = self.embedding(X,E)
        W = W.permute(1,0,2)
        W = W.double()
        output, state = self.rnn(W)
        return output # H: (L, N, D * H_{out})` when ``batch_first=False D = 2 if if 双向

class TriggerPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, trg_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,trg_size)
        
    def forward(self, H):
        '''
        input: H [L,B,D]
        
        output: T [B,L, trg_vocab_size],  T_pred [B,L]
        '''
        T = self.linear2(self.linear1(H))
        T = T.permute(1,0,2) # T 是输出
        T_pred = torch.argmax(T,dim = 2) # T_pred 是 对应概率最大的下标
        t = T_pred.reshape(-1) # 将预测的矩阵展平为[B*L]
        t = t.tolist()
        pred_trg_idx = [i for i in range(len(t)) if t[i] > 1]
        # 返回预测为trg的词的下标 列表形式
        return T, T_pred, pred_trg_idx
        

class ArgumentRolePrediction(nn.Module):
    def __init__(self, length,input_size, hidden_size, arg_size) -> None:
        super().__init__()
        # 此处的input_size因为前面双向，应该为两倍的hidden_size(有拼接), length 为 统一的 句子长度
        self.length = length
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,arg_size)
        
    def forward(self, H):
        H = H.permute(1,0,2) # [B,L,D]
        A_pred = []
        for j in range(self.length):
            h_ij = H[:,j,:] #[B,D]
            h_ij = h_ij.unsqueeze(-1).repeat(1,1,self.length)
            h_ij = h_ij.permute(0,2,1)
            # print('debug:',h_ij.shape)
            R_ij = torch.cat([H,h_ij],dim = 2)
            A_j = self.linear2(self.linear1(R_ij)) # [B,L,arg_vocab_size]
            A_pred.append(A_j)
        A_pred = torch.stack(A_pred,dim=0) # [L_j, B , L_i, arg_size]
        A_pred = A_pred.permute(1,2,0,3) # [B,L_i,L_j,arg_size]

        return A_pred


class JRNN(nn.Module):
    def __init__(self, embedding_weight, entity_type_dim, hidden_size, num_layers, trg_size, arg_size, length,device, dropout = 0, bidirectional = True,**kwargs):
        super().__init__()
        self.encoding_layer = EncodingLayer(embedding_weight,entity_type_dim,hidden_size,num_layers,dropout,bidirectional)
        self.trigger_prediction = TriggerPrediction(hidden_size *(2 if bidirectional else 1),hidden_size, trg_size)
        self.arg_prediction = ArgumentRolePrediction(length,2 * hidden_size *(2 if bidirectional else 1),hidden_size,arg_size)
        self.arg_size = arg_size
        self.length = length
        self.device = device
        
    def forward(self, X, E):
        device = self.device
        H = self.encoding_layer(X,E)
        T, T_pred, pred_trg_idx = self.trigger_prediction(H)
        A_pred = self.arg_prediction(H) # [B,L_i,L_j,arg_size]
        A_pred = A_pred.reshape([-1,A_pred.shape[2],A_pred.shape[3]]) # [B*L, L_j, arg_size]
        #A_pred = A_pred.to(device)
        E = E.reshape(-1) # B * L
        e = E.tolist()
        entity_idx = [i for i in range(len(e)) if e[i] > 1]
        other_idx = list( set(entity_idx) - set(pred_trg_idx) ) # 实体中没有被预测为trg的下标
        if len(other_idx) > 0:
            other = torch.tensor([1] + [0] * (self.arg_size-1)) # [1,0,0,...]
            other = other.to(device)
            other = other.repeat(self.length,1) # [L,arg_size]
            other = other.repeat(len(other_idx),1,1)
            # print('debug:', other.shape)
            # print('debug:', len(other_idx))
            A_pred[other_idx,:,:] = other.double()
        A_pred = A_pred[entity_idx] # 只留下了作为entity的argument_role predict
        return T, T_pred, pred_trg_idx, A_pred, entity_idx
    
    @torch.no_grad()
    def predict(self, X, E):
        