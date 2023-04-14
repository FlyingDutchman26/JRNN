import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import urllib.parse
import string
import re
import numpy as np
# ACE05Dataset
# 共17172条，句子最长154，平均15.6，长度超过50 的 仅有 458， 超过40的有 998
# 每行为一个字典，包含key如下：
# doc_id,sent_id,entity_mentions,relation_mentions,event_mentions,tokens,sentence
# 本任务所需内容:
# tokens: 列表: 已分词，区分大小写，包括了标点
# entity_mentions: [ {'id','start','end','entity_type','mention_type','text'} ]
# 经检查: entity_type 包括 {'FAC', 'GPE', 'VEH', 'LOC', 'PER', 'WEA', 'ORG'}, mention_type 只有 'UNK'

# event_mentions: [ {'event_type', 'id', 'trigger': {'start', 'end', 'text'}, 'arguments': [{'entity_id', 'text', 'role'}]} ]
# 这里的trigger长度都是1，意味着每个event_type都只会对应一个长度为1的触发词
# 8-33 subtype: {'Conflict:Attack', 'Justice:Arrest-Jail', 'Transaction:Transfer-Money', 'Life:Marry', 'Life:Injure', 'Business:Merge-Org', 'Business:Start-Org', 'Life:Die', 'Life:Divorce', 'Justice:Release-Parole', 'Personnel:End-Position', 'Justice:Sentence', 'Justice:Pardon', 'Justice:Convict', 'Life:Be-Born', 'Conflict:Demonstrate', 'Contact:Meet', 'Justice:Sue', 'Justice:Charge-Indict', 'Personnel:Start-Position', 'Contact:Phone-Write', 'Justice:Fine', 'Personnel:Nominate', 'Justice:Trial-Hearing', 'Justice:Appeal', 'Justice:Extradite', 'Business:End-Org', 'Justice:Acquit', 'Personnel:Elect', 'Movement:Transport', 'Business:Declare-Bankruptcy', 'Justice:Execute', 'Transaction:Transfer-Ownership'}
# 注意: event_mentions 中可能有多个event_type，对应多个事件 应该作为 label处理

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None: # reserved_tokens 如 <pad> <bos> <eos> <other>等, <unk>默认有
            reserved_tokens = []
       

        self.idx_to_token = ['<unk>'] + reserved_tokens # 前两个是 
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token in tokens:
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # token_to_index
        if not isinstance(tokens, (list, tuple)): # 如果接收到的tokens不是序列
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):  # Index for the unknown token
        return 0

class ACEDataset(Dataset):
    def __init__(self,data) -> None:
        super().__init__()
        self.sentence = data['sentence']
        self.entity = data['entity']
        self.trg = data['trigger']
        self.argument = data['arguments']
        self.valid_len = data['valid_len']
    
    def __getitem__(self, idx):
        return self.sentence[idx],self.entity[idx],self.trg[idx],self.argument[idx],self.valid_len[idx]
    
    def __len__(self):
        return len(self.sentence)

def read_data(root = './ace05E/', mode = 'train', max_len = 30, min_len = 3,num_examples = -1 ):
    # 取句子长度小于max_len的句子
    raw_data = pd.read_json(root + mode + '.oneie.json', lines=True)
    raw_data = raw_data.to_dict(orient='records')
    data = []
    for item in raw_data:
        if len(item['tokens']) <= max_len and len(item['tokens']) >= min_len:
            data.append(item)
    if num_examples > 0:
        data = data[:num_examples]
    return data

def tokenize_emb(entity_mentions,tokens):
    # 用 <unk> 代表不是实体
    emb_tokens = []
    valid_len = len(tokens)
    for item in entity_mentions:
        start = int(item['start'])
        end = int(item['end'])
        current_len = len(emb_tokens)
        emb_tokens += ['<unk>']*(start - current_len) + [item['entity_type']] * (end - start)
    emb_tokens += ['<unk>'] * (valid_len - len(emb_tokens))
    return emb_tokens

def tokenize_trigger(event_mentions,tokens):
    # 所有触发词都是长度1, 无所谓end
    # 用 <unk> 代表不是触发词 (other)
    trigger_tokens = []
    valid_len = len(tokens)
    for item in event_mentions:
        start = int(item['trigger']['start'])
        current_len = len(trigger_tokens)
        trigger_tokens += ['<unk>']*(start - current_len) + [item['event_type']]
    trigger_tokens += ['<unk>'] * (valid_len - len(trigger_tokens))
    return trigger_tokens

def tokenize_arg(event_mentions, entity_mentions, tokens):
    # 用 <unk> 代表 other, 即此word对于此trigger并不是argument
    # 对于不是trigger的词，它的arg预测a_ij我们到时候也一起用<unk>
    arg_tokens = []
    valid_len = len(tokens)
    for item in event_mentions:
        event_start = int(item['trigger']['start'])
        current_len = len(arg_tokens)
        # 对不是trigger的arg行全置为['<unk>']
        for i in range(event_start - current_len):
            arg_tokens.append(['<unk>'] * valid_len)
        arg_trg_tokens = [] # 构建对应trigger的arg行
        for arg in item['arguments']:
            arg_entity_id = arg['entity_id']
            arg_start = -1
            for entity in entity_mentions:
                if arg_entity_id == entity['id']:
                    arg_start = int(entity['start'])
                    break
                
            if arg_start < 0:
                raise('出错了，arg的id必须应该存在于entity的id中')
            
            current_len_arg = len(arg_trg_tokens)
            arg_trg_tokens += ['<unk>']*(arg_start - current_len_arg) + [arg['role']]
        arg_trg_tokens += ['<unk>'] * (valid_len - len(arg_trg_tokens))
        arg_tokens.append(arg_trg_tokens)
    for i in range(valid_len - len(arg_tokens)):
        arg_tokens.append(['<unk>'] * valid_len)
        
    return arg_tokens
                    
                    
        
def tokenize_data(data):
    # sentence 在数据集中已经tokenize好了,只需要改一下大小写，然后对其他数据tokenize
    tokenize_data = []
    for item in data:
        sentence_tokens = [text.replace('\u202f', ' ').replace('\xa0', ' ').lower() for text in item['tokens']]
        emb_tokens = tokenize_emb(item['entity_mentions'],item['tokens']) # 指entity 这里有写错
        trigger_tokens = tokenize_trigger(item['event_mentions'],item['tokens'])
        arg_tokens = tokenize_arg(item['event_mentions'],item['entity_mentions'],item['tokens'])
        tokenize_data.append({'sentence_tokens':sentence_tokens,'emb_tokens':emb_tokens,'trigger_tokens':trigger_tokens,'arg_tokens':arg_tokens})
    return tokenize_data

def truncate_pad(line, num_steps, padding_token = '<pad>'):
    """截断或填充序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def truncate_pad_data(data,max_length = 30):
    # 由于事件抽取任务要求完整句子，我一开始就筛掉了长句子，因此其实只有padding没有truncate
    pad_data = []
    for item in data:
        sentence_tokens = truncate_pad(item['sentence_tokens'],max_length)    
        emb_tokens = truncate_pad(item['emb_tokens'],max_length)
        trigger_tokens = truncate_pad(item['trigger_tokens'],max_length)
        arg_tokens = []
        for line in item['arg_tokens']:
            arg_tokens.append(truncate_pad(line,max_length))
        for i in range(max_length - len(arg_tokens)):
            arg_tokens.append(['<pad>']*max_length)
        pad_data.append({'sentence_tokens':sentence_tokens,'emb_tokens':emb_tokens,'trigger_tokens':trigger_tokens,'arg_tokens':arg_tokens})
    return pad_data


# 此部分处理word_embedding
def is_legal(s:str):
    if re.match('^[A-Za-z]+$', s):
        return True
    elif s.isnumeric():
        return True
    elif s in string.punctuation:
        return True
    elif s == ' ':
        return True
    else:
        return False

def process_embedding_data(device,data_path = './tencent-ailab-embedding-en-d100-v0.1.0-s'):
    f = open(data_path,encoding='utf-8', mode='r')
    raw_data = f.readlines()[1:] # 第一行为 2000000 100
    f.close()
    embd_data = []
    word_list = []
    for line in raw_data:
        line = line.split()
        word = urllib.parse.unquote(line[0])
        if is_legal(word):
            word_list.append(word)
            embd_data.append(line[1:])
    embd_weight = np.array(embd_data).astype(np.double)
    embd_weight = torch.tensor(embd_weight)
    unk_pad = torch.empty(2,100)
    unk_pad.normal_(embd_weight.mean(),embd_weight.var()) # 为<unk> 和 <pad> 构造embedding向量 
    embd_weight = torch.cat([unk_pad,embd_weight],dim=0)
    return word_list,embd_weight.double()

def build_array(raw_data, word_vocab:Vocab, ebd_vocab:Vocab, trg_vocab:Vocab, arg_vocab:Vocab,device):
    # 将data映射为整数值构建array
    sentence_array = []
    ebd_array = []
    trg_array = []
    arg_array = []
    for item in raw_data:
        sentence_array.append(word_vocab[item['sentence_tokens']])
        ebd_array.append(ebd_vocab[item['emb_tokens']])
        trg_array.append(trg_vocab[item['trigger_tokens']])
        arg_array.append(arg_vocab[item['arg_tokens']])
    sentence_array = torch.LongTensor(sentence_array)
    valid_len = (sentence_array != word_vocab['<pad>']).type(torch.int32).sum(1)
    return {'sentence':sentence_array.to(device),'entity':torch.LongTensor(ebd_array).to(device),'trigger':torch.LongTensor(trg_array).to(device),'arguments':torch.LongTensor(arg_array).to(device),'valid_len':valid_len.to(device)}

def load_array(data,batch_size,mode):
    '''返回dataloader迭代器'''
    is_train = (mode == 'train')
    dataset = ACEDataset(data)
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=is_train)

def build_vocab(word_reserved=['<pad>'],entity_reserved=['<pad>'],trg_reserved=['<pad>'],arg_reserved = ['<pad>'],device = 'cuda'):  
    word_list,embedding_weight = process_embedding_data(device)
    word_vocab = Vocab(word_list,reserved_tokens=word_reserved)
    entity_vocab = Vocab(['FAC', 'GPE', 'VEH', 'LOC', 'PER', 'WEA', 'ORG'],reserved_tokens=entity_reserved)
    trg_vocab = Vocab(['Conflict:Attack', 'Justice:Arrest-Jail', 'Transaction:Transfer-Money', 'Life:Marry', 'Life:Injure', 'Business:Merge-Org', 'Business:Start-Org', 'Life:Die', 'Life:Divorce', 'Justice:Release-Parole', 'Personnel:End-Position', 'Justice:Sentence', 'Justice:Pardon', 'Justice:Convict', 'Life:Be-Born', 'Conflict:Demonstrate', 'Contact:Meet', 'Justice:Sue', 'Justice:Charge-Indict', 'Personnel:Start-Position', 'Contact:Phone-Write', 'Justice:Fine', 'Personnel:Nominate', 'Justice:Trial-Hearing', 'Justice:Appeal', 'Justice:Extradite', 'Business:End-Org', 'Justice:Acquit', 'Personnel:Elect', 'Movement:Transport', 'Business:Declare-Bankruptcy', 'Justice:Execute', 'Transaction:Transfer-Ownership'],reserved_tokens=trg_reserved)
    arg_vocab = Vocab(['Person', 'Prosecutor', 'Recipient', 'Attacker', 'Artifact', 'Origin', 'Entity', 'Vehicle', 'Place', 'Seller', 'Agent', 'Defendant', 'Buyer', 'Giver', 'Beneficiary', 'Victim', 'Adjudicator', 'Target', 'Instrument', 'Org', 'Destination', 'Plaintiff'],reserved_tokens=arg_reserved)
    return word_vocab,entity_vocab,trg_vocab,arg_vocab,embedding_weight

def load_data(word_vocab,entity_vocab,trg_vocab,arg_vocab,device, batch_size = 30, max_len = 30 ,min_len = 3,num_examples = -1,root = './ace05E',mode = 'train'):
    pad_tokenized_data = truncate_pad_data(tokenize_data(read_data(root,mode,max_len,min_len,num_examples)),max_length=max_len)
    data = build_array(pad_tokenized_data,word_vocab,entity_vocab,trg_vocab,arg_vocab,device)
    dataloader = load_array(data,batch_size,mode)
    return dataloader

