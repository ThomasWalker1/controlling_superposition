import pickle
import os
import matplotlib.pyplot as plt

fig,ax=plt.subplots(nrows=1,ncols=1)
ax.set_xlabel("slope")
ax.set_ylabel("accuracy")
for file in os.listdir("from_scratch/accuracies_topological"):
    slope=float(file[0]+"."+file[1:])
    with open(f"from_scratch/accuracies_topological/{file}","rb") as accuracy_file:
        accuracies=pickle.load(accuracy_file)
    ax.plot(sorted(accuracies.keys()),[sum(accuracies[l])/len(accuracies[l]) for l in sorted(accuracies.keys())],color=plt.cm.viridis(float(file[0]+"."+file[1:])))
    ax.scatter(slope,sum(accuracies[slope])/len(accuracies[slope]),c="black")
plt.savefig("from_scratch/accuracies_topological.png",bbox_inches="tight")
