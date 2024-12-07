import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import pickle

import sys
sys.path.append("inceptionv1")

from utils.model import googlenet
from utils.data import *

DATASET_PATH="superposition/imagenet/sample"
BATCH_SIZE=32
SAMPLE_SIZE=4096

NEGATIVE_SLOPES=np.linspace(0,1,20)**2

INTERVENTION_LAYER="inception5a"

device="cpu"

dataset=TensorDataset("inceptionv1/sample/processed_tensors")

for neg_slope in NEGATIVE_SLOPES:
    
    neg_slope=round(neg_slope,ndigits=4)

    OUTPUTS_PATH=f"inceptionv1/intervention/{INTERVENTION_LAYER}/{str(neg_slope).replace('.','')}"
    if not(os.path.exists(OUTPUTS_PATH)):
        os.makedirs(OUTPUTS_PATH)

    model=googlenet(weights="GoogLeNet_Weights.DEFAULT", neg_slope=neg_slope)

    loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False)

    model.eval()

    if os.path.exists(f"{OUTPUTS_PATH}/model_accuracies"):
        with open(f"{OUTPUTS_PATH}/model_accuracies","rb") as file:
            model_accuracies=pickle.load(file)
        last_tested_id=len(model_accuracies.keys())-1
    else:
        model_accuracies={}
        last_tested_id=-1

    if last_tested_id==4095:
        continue

    with torch.no_grad():
        img_idx=0
        for images,labels in tqdm(loader,desc=f"evaluating {neg_slope}",total=SAMPLE_SIZE//BATCH_SIZE+(1-int(SAMPLE_SIZE%BATCH_SIZE==0))):
            
            if last_tested_id<=img_idx+BATCH_SIZE:
                # images=images.to(device)
                # labels=labels.to(device)

                output=model(images)[0]

                predicted_logits,predicted_labels=torch.max(output,1)

                correct_mask=(predicted_labels==labels)

                for i in range(len(labels)):
                    if not(img_idx in model_accuracies.keys()):
                        model_accuracies[img_idx]={
                            "predicted_label":predicted_labels[i].item(),
                            "predicted_class":get_class(int(predicted_labels[i])),
                            "predicted_confidence":torch.max(softmax(output[i,:],dim=0)).item(),
                            "actual_label":labels[i].item(),
                            "actual_class":get_class(int(labels[i])),
                            "actual_confidence":softmax(output[i,:],dim=0)[labels[i]].item()
                        }
                    img_idx+=1
            else:
                img_idx+=BATCH_SIZE
            
            with open(f"{OUTPUTS_PATH}/model_accuracies","wb") as file:
                pickle.dump(model_accuracies,file)