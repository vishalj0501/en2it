import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embed = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        x = x.to(torch.long)
        return self.embed(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model : int ,seq_len: int,dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        # pe = pe.unsqueeze(0).transpose(0,1)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
     
    def forward(self,x):
        # x=x+torch.tensor(self.pe[:,:x.size(1)],requires_grad=False)
        x = x+(self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    


    def forward(self, x):
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :]  # Adjust the positional encoding size
        x = x + pe_slice.requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps: float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #x 
        self.beta = nn.Parameter(torch.zeros(1)) #+

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.beta 
    
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model,d_ff) #w1 & b1
        self.linear2 = nn.Linear(d_ff,d_model) #w2 & b2 
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model : int, h : int, dropout : float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model % h != 0"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model) #w_q
        self.w_k = nn.Linear(d_model,d_model) #w_k
        self.w_v = nn.Linear(d_model,d_model) #w_v
        self.w_o= nn.Linear(d_model,d_model) #w_o
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query,key,value,mask,dropout=nn.Dropout):
        d_k = query.shape[-1]
        attention_score = torch.matmul(query,key.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0,-1e9)
        attention_score = torch.softmax(attention_score,dim=-1)

        if dropout is not None:
            attention_score = dropout(attention_score)
        
        return (attention_score@value),attention_score


    def forward(self,q,k,v,mask=None):
        query = self.w_q(q) #
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        
        # here, view is used to split the d_model into h heads, and transpose is used to swap the seq_len and head dimensions

        x,self.attention_score = self.attention(query,key,value,mask,self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self,features:int,dropout:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features) 

    def forward(self,x,sublayer):
        sub_layer_norm= sublayer(self.norm(x))
        return x + self.dropout(sub_layer_norm) # the paper does sublayer(x) , and then norm
    

class EncoderBlock(nn.Module):
    def __init__(self,features:int,self_attention_block: MultiHeadAttention,feed_forward_block: FeedForward,dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for i in range(2)])
    
    def forward(self,x,src_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        return self.residual_connection[1](x,self.feed_forward_block)
        
    

class Encoder(nn.Module):
    def __init__(self,features: int,layers: nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm= LayerNormalization(features)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
     

class DecoderBlock(nn.Module):
    def __init__(self,features:int ,self_attention_block: MultiHeadAttention, cross_attention_block:MultiHeadAttention,feed_forward:FeedForward,dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(features,dropout) for i in range(3)])

    def forward(self,x,enc_output,source_mask,target_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention_block(x,x,x,target_mask))
        x = self.residual_connection[1](x,lambda x: self.cross_attention_block(x,enc_output,enc_output,source_mask))
        x = self.residual_connection[2](x,self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self,features:int ,layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self,x,enc_output,source_mask,target_mask):
        for layer in self.layers:
            x = layer(x,enc_output,source_mask,target_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    def __init__(self,d_model: int,vocab_size: int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    


class Transformer(nn.Module):
    def __init__(self,encoder: Encoder,decoder: Decoder, src_embed: InputEmbeddings,target_embed:InputEmbeddings, src_pos:PositionalEncoding, target_pos:PositionalEncoding,projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        return self.encoder(self.src_pos(self.src_embed(src)),src_mask)
    
    def decode(self,enc_output,src_mask,target,target_mask):
        # return self.decoder(self.target_pos(self.target_embed(target)),enc_output,src_mask,target_mask)
        target=self.target_embed(target)
        target=self.target_pos(target)
        return self.decoder(target,enc_output,src_mask,target_mask)

    def project(self,x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int,target_vocab_size:int,src_seq_len:int,target_seq_len:int ,d_model:int = 512, N: int =6,h:int=8 , dropout:float = 0.1,d_ff:int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    target_embed = InputEmbeddings(d_model,target_vocab_size)

    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    target_pos = PositionalEncoding(d_model,target_seq_len,dropout)

    projection_layer = ProjectionLayer(d_model,target_vocab_size)

    # encoder = Encoder(d_model,nn.ModuleList([EncoderLayer(MultiHeadAttention(d_model,h,dropout),FeedForward(d_model,d_ff,dropout),dropout) for i in range(N)]))
    # decoder = Decoder(d_model,nn.ModuleList([DecoderBlock(MultiHeadAttention(d_model,h,dropout),MultiHeadAttention(d_model,h,dropout),FeedForward(d_model,d_ff,dropout),dropout) for i in range(N)]))

    encoder_blocks=[]
    decoder_blocks=[]
    for i in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    for i in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    transformer = Transformer(encoder,decoder,src_embed,target_embed,src_pos,target_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer






