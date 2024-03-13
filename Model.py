from torch.nn import Module, Linear, functional


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = Linear(13, 10)
        self.fc2 = Linear(10, 2)

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
