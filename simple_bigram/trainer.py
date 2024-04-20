#!/bin/env python3
from .model import *
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("./data/data.txt") as f:
    text = f.read()[:20000]
chars= sorted(set(text))
# print(len(chars))

chars = set(text)
tktx = dict(enumerate(chars))
txtk = dict(zip(tktx.values(),tktx.keys()))

# print(txtk)
# print(tktx)

encode = lambda x: [ txtk[i] for i in x]
decode = lambda x: "".join(tktx[i] for i in x)

# print(x:=encode(text[:200]))
# print(decode(x))


def main():
    print("--> main()")
    model = BiGram( vocab_size= len(chars)).to(device=device)
    # while True:...
    print("<-- main()")
    


if __name__=='__main__':
    main()