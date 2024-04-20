import torch
import simple_bigram


def test_model():
    'load model to cuda'
    simple_bigram.model.BiGram(80).to('cuda')

