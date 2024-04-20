import torch
import torch.nn as nn
import dataclasses as d
# import torch.nn.functional as F
# import simple_bigram as sb

@d.dataclass
class ModelConfig:
    d_model,
    vocab_size

class FeedForward(nn.Module):

    def __init__( self , in_layer : int , d_model: int ) -> None:
        super().__init__()
        self.d_model = d_model
        self.inp = nn.Linear(d_model , d_model//2 , bias=True)
        self.l1 = nn.Linear(d_model//2, d_model//2 , bias=True)
        self.l2 = nn.Linear(d_model//2, d_model//2 , bias=True)
        # self.l3 = nn.Linear(d_model//2 , d_model)
        self.out = nn.Linear(d_model//2 , d_model , bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        x = x.view(-1 , self.in_layer )
        x = self.relu(self.inp(x))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out(x)
        return x


class BiGram(nn.Module):

    def __init__( self , vocab_size: int ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_table = nn.Embedding( vocab_size, 3)
        self.layer_norm = nn.LayerNorm( normalized_shape=3, bias=True)
        self.feedforward = FeedForward(in_layer= vocab_size*3 ,d_model=vocab_size)
        self.softmax = nn.Softmax( dim=-1 )

    def forward( self , x: torch.Tensor ) -> torch.Tensor :
        x = x.view( -1, self.vocab_size )
        x = self.feedforward( x )
        return self.softmax( x )

