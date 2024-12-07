import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
from tqdm import tqdm
import pickle

import os

def sample_from_ball(n=100, d=2, r=1, seed=None):
    rng = np.random.default_rng(seed)

    U = rng.normal(size=(n, d + 2))
    norms = np.sqrt(np.sum(np.abs(U) ** 2, axis=-1))
    U = r * U / norms[:, np.newaxis]
    X = U[:, 0:d]

    return np.asarray(X)

def sample_from_annulus(n, r, R, d=2, seed=None):
    if r >= R:
        raise RuntimeError(
            "Inner radius must be less than or equal to outer radius"
        )

    rng = np.random.default_rng(seed)
    if d == 2:
        thetas = rng.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(rng.uniform(r**2, R**2, n))
        X = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    else:
        X = np.empty((0, d))
        while True:
            sample = sample_from_ball(n, d, r=R, seed=rng)
            norms = np.sqrt(np.sum(np.abs(sample) ** 2, axis=-1))
            X = np.row_stack((X, sample[norms >= r]))
            if len(X) >= n:
                X = X[:n, :]
                break
    return X

TRAIN_SAMPLE_SIZE=512
TEST_SAMPLE_SIZE=int(0.2*TRAIN_SAMPLE_SIZE)
BATCH_SIZE=32
INPUT_DIM=3

X_train=np.row_stack([sample_from_ball(n=TRAIN_SAMPLE_SIZE,d=INPUT_DIM,r=1),sample_from_annulus(n=TRAIN_SAMPLE_SIZE,r=1.5,R=2.5,d=INPUT_DIM)])
Y_train=np.concatenate([np.ones(TRAIN_SAMPLE_SIZE),np.zeros(TRAIN_SAMPLE_SIZE)])

X_test=np.row_stack([sample_from_ball(n=TEST_SAMPLE_SIZE,d=INPUT_DIM,r=1),sample_from_annulus(n=TEST_SAMPLE_SIZE,r=1.5,R=2.5,d=INPUT_DIM)])
Y_test=np.concatenate([np.ones(TEST_SAMPLE_SIZE),np.zeros(TEST_SAMPLE_SIZE)])

train_data=data.TensorDataset(torch.tensor(X_train,dtype=torch.float32),torch.tensor(Y_train,dtype=torch.long))
test_data=data.TensorDataset(torch.tensor(X_test,dtype=torch.float32),torch.tensor(Y_test,dtype=torch.long))


train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)


class FC(nn.Module):
    def __init__(self,negative_slope):
        super().__init__()

        self.features=nn.Linear(INPUT_DIM,32)
        self.classifier=nn.Linear(32,2)
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

        model=train(model,train_iterator,optimizer,criterion,50,"cpu")
        state_dict=model.state_dict()
        accuracies=test_model(model,state_dict)
        if os.path.exists(f"from_scratch/accuracies_topological/{str(base_slope).replace('.','')}"):
            with open(f"from_scratch/accuracies_topological/{str(base_slope).replace('.','')}","rb") as file:
                current_accuracies=pickle.load(file)
                for (k,v) in accuracies.items():
                    current_accuracies[k].append(v[0])
            with open(f"from_scratch/accuracies_topological/{str(base_slope).replace('.','')}","wb") as file:
                pickle.dump(current_accuracies,file)
        else:
            with open(f"from_scratch/accuracies_topological/{str(base_slope).replace('.','')}","wb") as file:
                pickle.dump(accuracies,file)