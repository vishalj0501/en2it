import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embed = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embed(x)*torch.sqrt(torch.tensor(self.d_model,dtype=torch.float32))
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,seq_len,dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len-1,dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        # pe = pe.unsqueeze(0).transpose(0,1)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
     
    def forward(self,x):
        x=x+torch.tensor(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)