import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

    def build(self, input_shape):
        raise NotImplementedError
    
    def train(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
