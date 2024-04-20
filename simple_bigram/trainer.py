#!/bin/env python3
from .model import *
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tktx = dict(enumerate(chars))
txtk = dict(zip(tktx.values(),tktx.keys()))


encode = lambda x: [ txtk[i] for i in x]
decode = lambda x: "".join(tktx[i] for i in x)

