import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from dataclasses import dataclass

from tqdm import trange

@dataclass
class Config:
  c_id: int
  n_features: int
  n_hidden: int
  n_instances: int

  negative_slope: float
  feature_probability: float

def save_config(config):
  with open(f"toy_models/configs/{config.c_id}","wb") as file:
    pickle.dump(config,file)
 
class Model(nn.Module):
  def __init__(self,config,importance,device='cpu'):
    super().__init__()
    self.config = config
    self.W = nn.Parameter(torch.empty((config.n_features, config.n_hidden), device=device))
    nn.init.xavier_normal_(self.W)
    self.b_final = nn.Parameter(torch.zeros(config.n_features, device=device))

    self.negative_slope = config.negative_slope

    self.feature_probability = config.feature_probability
    self.importance = importance.to(device)

  def forward(self, features):
    
    hidden = torch.einsum("...f,fh->...h", features, self.W)
    out = torch.einsum("...h,fh->...f", hidden, self.W)
    out = out + self.b_final
    out = F.relu(out) - self.negative_slope * F.relu(-out)
    # out = F.relu(out)
    return out, hidden

  def generate_batch(self, n_batch):
    feat = torch.rand((n_batch, self.config.n_features), device=self.W.device)
    batch = torch.where(
        torch.rand((n_batch, self.config.n_features), device=self.W.device) <= self.feature_probability,
        feat,
        torch.zeros((), device=self.W.device),
    )
    return batch
  
def linear_lr(step, steps):
  return (1 - (step / steps))

def constant_lr(*_):
  return 1.0

def cosine_decay_lr(step, steps):
  return np.cos(0.5 * np.pi * step / (steps - 1))

def optimize(model,n_batch=1024,steps=10_000,print_freq=100,lr=1e-3,lr_scale=constant_lr,hooks=[]):

  opt = torch.optim.AdamW(list(model.parameters()), lr=lr)

  with trange(steps) as t:
    for step in t:
      step_lr = lr * lr_scale(step, steps)
      for group in opt.param_groups:
        group['lr'] = step_lr
      opt.zero_grad(set_to_none=True)
      batch = model.generate_batch(n_batch)
      out = model(batch)[0]
      error = (model.importance*(batch.abs() - out)**2)
      loss = einops.reduce(error, 'b f -> 1', 'mean').sum()
      loss.backward()
      opt.step()
    
      if hooks:
        hook_data = dict(model=model,step=step,opt=opt,error=error,loss=loss,lr=step_lr)
        for h in hooks:
          h(hook_data)
      if step % print_freq == 0 or (step + 1 == steps):
        t.set_postfix(loss=loss.item(),lr=step_lr)

def plot_instances(c_id):

  model_data={float(slope.split(".")[0][0]+"."+slope.split(".")[0][1:]):
              {
                "W":torch.load(f"toy_models/state_dicts_{c_id}/{slope}",weights_only=True)["W"],
                "b":torch.load(f"toy_models/state_dicts_{c_id}/{slope}",weights_only=True)["b_final"],
                "importance":torch.load(f"toy_models/importance_{c_id}/{slope}",weights_only=True),
                "activations":torch.load(f"toy_models/activations_{c_id}/{slope}",weights_only=True).detach().numpy()
                } for slope in os.listdir(f"toy_models/state_dicts_{c_id}")}
  
  from matplotlib import colors  as mcolors
  from matplotlib import collections  as mc

  plt.rcParams['figure.dpi'] = 200
  fig, axs = plt.subplots(2,len(model_data.keys()),figsize=(2*len(model_data.keys()),4))

  for slope,ax in zip(sorted(model_data.keys()),axs[0,:]):

    ax.set_title(str(slope))
    W=model_data[slope]["W"]
    b=model_data[slope]["b"]
    importance=model_data[slope]["importance"]

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(importance.cpu().numpy()))
      
    colors = [mcolors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    ax.scatter(W[:,0], W[:,1], c=colors[0:len(W[:,0])],zorder=0)
    ax.set_aspect('equal')
    ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W),W), axis=1), colors=colors, zorder=0))
    
    z = 1.5
    ax.set_facecolor('#FCFBF8')
    ax.set_xlim((-z,z))
    ax.set_ylim((-z,z))
    ax.tick_params(left=True,right=False,labelleft=False,labelbottom=False,bottom=True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_position('center')
  
  for slope,ax in zip(sorted(model_data.keys()),axs[1,:]):
    z=1.5
    ax.set_xlim((-z,z))
    ax.set_ylim((-z,z))

    W=model_data[slope]["W"]
    b=model_data[slope]["b"]
    
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)

    points=np.vstack([X.ravel(),Y.ravel()])
    zs=(W@points+b[:,np.newaxis]).detach().numpy()
    zs=np.clip(zs,0,zs.max())-slope*np.clip(-zs,0,(-zs).max())
    gs=np.ones_like(zs)
    gs[np.where(zs<=0)]=-slope*gs[np.where(zs<=0)]
    columns,column_ids=np.unique(gs,return_inverse=True,axis=1)

    ax.scatter(points[0,:],points[1,:],c=column_ids,s=0.5)
    ax.axis("off")

    for k in range(np.shape(W)[0]):
      xs=np.linspace(-1.5,1.5,100)
      ys=(-b[k]-W[k,0]*xs)/(W[k,1])
      ax.plot(xs,ys,color="black",linewidth=1)


    plt.savefig(f"toy_models/{c_id}.png")