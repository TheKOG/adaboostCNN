__author__ = 'KOG'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from tqdm import tqdm
import pdb

class BaselineModel(nn.Module):
    def __init__(self, n_features=10, n_classes=3,seed=100):
        super(BaselineModel, self).__init__()
        self.n_features=n_features
        self.n_classes=n_classes
        self.seed=seed
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(in_features=32*n_features//2, out_features=128),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=n_classes),
            nn.Softmax(dim=1)
        ).to(torch.float32)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adagrad(self.parameters())
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # for name, child in self.model.named_children():
        #     print(name, child)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
        import pdb
        # pdb.set_trace()
        if(x.shape[2]!=self.n_features):
            x = x.permute(0, 2, 1)
        return self.model(x)

    def fit(self, X, y_b, epochs, batch_size, sample_weight=None):
        X = torch.tensor(X, dtype=torch.float32)
        y_b = torch.tensor(y_b, dtype=torch.float32)
        dataset = TensorDataset(X, y_b)
        if sample_weight is not None:
            sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(epochs)):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, X, y):
        self.eval()
        y=torch.tensor(y,dtype=torch.float32)
        with torch.no_grad():
            output = self(X)
            loss = self.criterion(output, y.to(torch.float32))
            _, predicted = torch.max(output.data, 1)
            _,y=torch.max(y.data, 1)
            # pdb.set_trace()
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)
        return [loss.item(), accuracy]
    
    def predict(self,x):
        return self(x).cpu().detach().numpy()