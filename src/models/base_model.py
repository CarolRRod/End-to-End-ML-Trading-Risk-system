class BaseModel:
    def __init__(self, config=None):
        self.config = config or {}

    def build(self, input_shape):
        raise NotImplementedError
    
    def train(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError