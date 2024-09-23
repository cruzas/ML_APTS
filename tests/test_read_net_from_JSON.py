import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json
import os
import sys



def concat_fun(out_1, out_2):
    return torch.cat([out_1, out_2])

# First number is the stage number, second number (after '.') means that there are stages running in parallel
# E.g. stage_3.X means that stage 3 is run in parallel

def preprocessing(x):
    #returns the first 700 features
    return x[:, :700], x[:, -84:]

def preprocessing2(x):
    #returns the last 84 features
    return x[:, -84:]

# NOTE: It is mandatory to have "start" and "finish" keys in the dictionary. The rank which owns the "start" layer will be the rank that receives the input data from the data loader, same goes for the "finish" layer which receives the target data from the data loader.
net2 = {
    "start": {
        "callable": {'object': preprocessing, 'settings': {}},
        "dst": {"to": ["input_layer1", "input_layer2"], "strategy": None},
        "rcv": {"src": None, "strategy": None},
        "stage": 1,
    },
    "input_layer1": {
        "callable": {'object': nn.Linear, 'settings': {"in_features": 700, "out_features": 256}},
        "dst": {'to': ["stage2_layer1"], 'strategy': None}, # strategy must have as many outputs as the length of the to list (input will be the output of the processed data in current callable)
        "rcv": {'src': ["start"], 'strategy': None},
        "stage": 1.1, # First number: stage; second number: process number
    },
    "input_layer2": {
        "callable": {'object': nn.Linear, "settings": {"in_features": 84, "out_features": 256}},
        "dst": ["stage2_layer1"],
        "rcv": {"src": ["preprocessing1"], "strategy": None},
        "stage": 1.2,
    },    
    "finish": {}
}




def make_network_from_dict(net_dict):
    # Create a dictionary of all the layers
    pass 


def main():
    print("hello world")




if __name__ == "__main__":
    main()
