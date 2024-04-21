# import torch
import torch
from simple_bigram.model import BiGram, ModelConfig


def test_model_to_cuda():
    'load model to cuda'
    torch.set_default_device('cuda')
    config = ModelConfig(
        d_model=80,
        vocab_size=80
    )
    model = BiGram(config=config)
    x = model(torch.randn(1,80))
    print(x.shape)

def test_model():
    'load model to cpu'
    torch.set_default_device('cpu')
    config = ModelConfig(
        d_model=80,
        vocab_size=80
    )
    model = BiGram(config=config)
    x = model(torch.randn(1,80))
    print(x.shape)
