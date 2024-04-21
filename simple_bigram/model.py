import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    vocab_size: int

class FeedForward(nn.Module):

    def __init__( self , d_model: int ) -> None:
        super().__init__()
        self.d_model = d_model
        self.inp = nn.Linear(d_model , d_model//2 , bias=True)
        self.l1 = nn.Linear(d_model//2, d_model//2 , bias=True)
        self.l2 = nn.Linear(d_model//2, d_model//2 , bias=True)
        # self.l3 = nn.Linear(d_model//2 , d_model)
        self.out = nn.Linear(d_model//2 , d_model , bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor ) -> torch.Tensor:
        x = x.view(-1 , self.d_model )
        x = self.relu(self.inp(x))
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.out(x)
        return x


class BiGram(nn.Module):

    def __init__( self , config:ModelConfig ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.config = config
        self.embedding_table = nn.Embedding( self.vocab_size, 3)
        self.layer_norm = nn.LayerNorm( normalized_shape=3, bias=True)
        self.feedforward = FeedForward(d_model=self.d_model)
        self.softmax = nn.Softmax( dim=-1 )

    def forward( self , x: torch.Tensor ) -> torch.Tensor :
        x = x.view( -1, self.vocab_size )
        x = self.feedforward( x )
        return self.softmax( x )

