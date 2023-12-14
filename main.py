# coding: utf-8

import json
from train import train
from eval3 import eval

with open(r'train_config.json', encoding="utf-8") as f:
    config = json.load(f)
flag = "3band1"
train(config,flag)
with open(r'eval_config3.json', encoding='utf-8') as f:
    config = json.load(f)
eval(config,flag)
print(flag,"-----------------------------------------------------------------------------------")




