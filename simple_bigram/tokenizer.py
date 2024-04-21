#!/bin/env python3
from .model import *
import torch
import torch.nn as nn

# import tokenizers


# device = "cuda" if torch.cuda.is_available() else "cpu"


class Tokenizer:
    def __call__(self, x):
        return self.encoder(x)

    def __init__(self, dataset: str) -> None:
        with open(dataset) as F:
            text = F.read()
        self.chars = set(text)
        self.vocab_size = len(self.chars)
        self.tktx = dict(enumerate(self.chars))
        self.txtk = dict(zip(self.tktx.values(), self.tktx.keys()))

    def encoder(self, x):
        return [self.txtk[i] for i in x]

    def decoder(self, x):
        return "".join(self.tktx[i] for i in x)
