import numpy as np
import torch
import networkx as nx
from matplotlib import pyplot as plt
from burgers import Burgers

# 暂时没用
def load_env(env_name):
    if env_name == 'burger':
        return Burgers()
    else:
        raise ValueError('Invalid environment name')


