import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer
from dataclasses import dataclass
# '''注意力计算函数'''
# def attention(query,key,value,dropout=None):
#     '''
#     args:
#     query:查询值矩阵
#     key:键值矩阵
#     value:真值矩阵
#     '''
#     # 获取键向量的维度，键向量的维度和值向量的维度相同
#     d_k = query.size(-1)
#     # 计算Q与K的内积并除以根号dk
#     #transpose(-2,-1)表示对最后两个维度进行转置（即将最后一维和倒数第二维交换位置）
#     scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
#     # Softmax
#     p_attn = scores.softmax(dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     # 根据计算结果对value进行加权求和
#     return torch.matmul(p_attn,value),p_attn

@dataclass
class ModelArgs:
    n_embd:int #嵌入维度
    n_heads:int #注意力头数
    dim:int #模型维度
    dropout:float
    max_seq_len:int
    vocab_size:int
    block_size:int
    n_layer:int

class MultiHeadAttention(nn.Module):

    def __init__(self,args:ModelArgs,is_causal=False):
        super().__init__()
        assert args.dim%args.n_heads == 0
        self.head_dim = args.dim//args.n_heads
        self.n_heads = args.n_heads

        #Wq,Wk,Wv参数矩阵，每个参数矩阵为 n_embd*dim
        self.wq = nn.Linear(args.n_embd,args.n_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(args.n_embd,args.n_heads*self.head_dim,bias=False)
        self.wv = nn.Linear(args.n_embd,args.n_heads*self.head_dim,bias=False) 
        #输出权重矩阵，维度为 dim*dim
        self.wo =nn.Linear(args.dim,args.dim,bias=False)
        #注意力dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        #残差连接dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        if is_causal:
            mask = torch.full((1,1,args.max_seq_len,args.max_seq_len),float('-inf'))
            mask = torch.triu(mask,diagonal=1)
            self.register_buffer('mask',mask)
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor):  
        bsz,seqlen,_ = q.size()

        xq,xk,xv = self.wq(q),self.wk(k),self.wv(v)
        #将Q、K、V拆分成多头
        xq,xk,xv = xq.view(-1,seqlen,self.n_heads,self.head_dim),xk.view(-1,seqlen,self.n_heads,self.head_dim),xv.view(-1,seqlen,self.n_heads,self.head_dim)
        xq,xk,xv =xq.transpose(1,2),xk.transpose(1,2),xv.transpose(1,2)
        #scores:[bsz,n_heads,seqlen,seqlen]
        scores = torch.matmul(xq,xk.transpose(-1,-2))/math.sqrt(self.head_dim)
        #掩码子注意力必须有注意力掩码
        if self.is_causal:
            scores = scores + self.mask[:,:,:seqlen,:seqlen]
        scores = F.softmax(scores.float(),dim=-1)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores,xv)
        output = output.transpose(1,2).contiguous().view(bsz,seqlen,-1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class LayerNorm(nn.Module):
    '''LayerNorm层'''
    def __init__(self,features,eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  

    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self,dim: int,hidden_dim: int,dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.fc2(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.attention_norm = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args,is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.ffn = MLP(args.n_embd,args.dim*4,args.dropout)

    def forward(self,x):
        # Layer Norm
        x = self.attention_norm(x)
        #自注意力
        h = x+self.attention(x,x,x)
        #前馈神经网络
        out = h+self.ffn(self.ffn_norm(h))
        return out

class Encoder(nn.Module):
    '''Encoder块'''
    def __init__(self,args):
        super().__init__()
        #一个Encoder由N个EncoderLayer组成,先用列表推导式构建N个EncoderLayer，后用EncoderLayer将网络层注册到模型中
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm =  LayerNorm(args.n_embd)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.attention_norm_1 = LayerNorm(args.n_embd)
        self.mask_attention = MultiHeadAttention(args,is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        self.attention = MultiHeadAttention(args,is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        self.ffn = MLP(args.dim,args.dim,args.dropout)

    def forward(self,x,enc_out):
        x = self.attention_norm_1(x)
        x = x+self.mask_attention(x,x,x)
        x = self.attention_norm_2(x)
        h = x+self.attention(x,enc_out,enc_out)
        out = h+self.ffn(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self,x,enc_out):
        for layer in self.layers:
            x = layer(x,enc_out)
        return self.norm(x)

class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)]
        return x

class Transformer(nn.Module):
    def __init__(self,args):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        # apply()是Pytorch的nn.Module提供的一个方法，作用是：
        # 递归遍历当前神经网络中的所有子模块，并对每个模块执行指定的函数。
        self.apply(self._init_weights)
        print("number of parameters:%.2fM"%(self.get_num_params()/1e6))

    '''统计所有参数的数量'''

    def get_num_params(self,non_embedding=False):
        # non_embedding=True表示不统计Embedding层的参数

        # p.numel() for p in self.parameters()生成器表达式
        # p.numel()是Pytorch张量的方法，意思是返回张量中元素的总数量
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= sum(p.numel() for p in self.transformer.wte.parameters())
        return n_params

    '''初始化权重'''

    def _init_weights(self,module):
        #isinstance()前查后，判断前面的对象是不是后面的类型
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    '''前向计算函数'''

    def forward(self,idx,targets=None):
        device = idx.device
        b,t = idx.size()
        assert t<=self.args.block_size,f"不能计算该序列，该序列长度为{t},最大长度为{self.args.block_size}"
        print("idx",idx.size())
        #通过Embedding层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        #然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb)
        #然后再进行Dropout
        x = self.transformer.drop(pos_emb)
        #然后进行Encoder
        print("x after wpe:", x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:", enc_out.size())
        #再通过Decoder
        x = self.transformer.decoder(x,enc_out)
        print("x after decoder:", x.size())

        if targets is not None:
            #训练阶段，如果我们给了target，就计算loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
            return loss
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


def main():
    args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 1000, 2)
    text = "我喜欢快乐地学习大模型"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_seq_len,
        truncation=True,
        padding='max_length'
    )
    args.vocab_size = tokenizer.vocab_size
    transformer = Transformer(args)
    inputs_id = inputs_token['input_ids']
    logits, loss = transformer.forward(inputs_id)
    print(logits)
    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode(predicted_ids)
    print("预测结果:", output)

if __name__ == "__main__":
    print("开始")
    main()