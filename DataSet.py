from torch.utils.data import Dataset as TorchDataSet
from torch import Tensor
from pandas import read_csv, to_numeric


class DataSet(TorchDataSet):
    def __init__(self, path):
        self.data = read_csv(path)

        self.data.loc[self.data['stabf'] == 'unstable', 'stabf'] = 0  # 0 -> unstable
        self.data.loc[self.data['stabf'] == 'stable', 'stabf'] = 1  # 0 -> stable

        self.data = self.data.apply(to_numeric)

        self.data = self.data.values

        self.X = Tensor(self.data[:, :13]).float()
        self.Y = Tensor(self.data[:, 13]).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
