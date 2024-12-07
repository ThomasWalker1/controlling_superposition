import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm
import pickle

import os

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images (mean and std for MNIST)
])

ROOT="from_scratch/data"

train_data = datasets.MNIST(ROOT,
                              train=True,
                              download=True,
                              transform=transform)

test_data = datasets.MNIST(ROOT,
                             train=False,
                             download=True,
                             transform=transform)

BATCH_SIZE = 128

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

class FC(nn.Module):
    def __init__(self,negative_slope):
        super().__init__()

        self.features=nn.Linear(784,8)
        self.classifier=nn.Linear(8,10)
        self.negative_slope=negative_slope

    def forward(self, x):
        x=torch.flatten(x,start_dim=1)
        x=self.features(x)
        x=F.leaky_relu(x,negative_slope=self.negative_slope)
        x=self.classifier(x)
        return x
    
def initialize_parameters(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def evaluate(model, iterator, device):
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            acc = calculate_accuracy(y_pred, y)
            epoch_acc += acc.item()
    return epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion, epochs, device):
    model.train()
    pbar=tqdm(range(epochs))
    for _ in pbar:
        for step,(x, y) in tqdm(enumerate(iterator), desc="Training", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if step%25==0:
                pbar.set_description(f"{loss.item():.4f}")
    return model
    
NEGATIVE_SLOPES=[0.0,0.05,0.15,0.25,0.4,0.6,0.75,0.9]
SAMPLES=3

def test_model(model,state_dict):
    accuracies={}
    for negative_slope in NEGATIVE_SLOPES:
        model=FC(negative_slope)
        model.load_state_dict(state_dict)
        accuracies[negative_slope]=[evaluate(model,test_iterator,device="cpu")]
    return accuracies


for sample in range(SAMPLES):
    for base_slope in NEGATIVE_SLOPES:
        print(f"sample {sample} - base slope {base_slope}")
        model=FC(base_slope)
        model.apply(initialize_parameters)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        device="cpu"

        model=train(model,train_iterator,optimizer,criterion,10,"cpu")
        state_dict=model.state_dict()
        accuracies=test_model(model,state_dict)
        if os.path.exists(f"from_scratch/accuracies/{str(base_slope).replace('.','')}"):
            with open(f"from_scratch/accuracies/{str(base_slope).replace('.','')}","rb") as file:
                current_accuracies=pickle.load(file)
                for (k,v) in accuracies.items():
                    current_accuracies[k].append(v[0])
            with open(f"from_scratch/accuracies/{str(base_slope).replace('.','')}","wb") as file:
                pickle.dump(current_accuracies,file)
        else:
            with open(f"from_scratch/accuracies/{str(base_slope).replace('.','')}","wb") as file:
                pickle.dump(accuracies,file)