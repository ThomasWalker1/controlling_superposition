import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
import numpy as np

capacities={"inception3a":256,
            "inception3b":480,
            "inception4a":512,
            "inception4b":512,
            "inception4c":512,
            "inception4d":528,
            "inception4e":832,
            "inception5a":832,
            "inception5b":1024}

def line_area(xs,ys):
    area=0
    if len(xs)==1:
        return area
    for k in range(len(xs)-1):
        area+=(xs[k+1]-xs[k])*(ys[k]+ys[k+1])/2
    return area

def line_variation(xs,ys):
    diffs=0
    for k in range(len(ys)-1):
        diffs+=abs((ys[k+1]-ys[k])/(xs[k+1]-xs[k]))
    return diffs

def power_law(x,a,b):
    return a* x**b

def layer_specific(intervention_layer):
    OUTPUTS_PATH=f"inceptionv1/intervention/{intervention_layer}"
    tested_slopes={float(dir[0]+"."+dir[1:]):dir for dir in os.listdir(OUTPUTS_PATH) if "0" in dir}

    accuracies_across_slopes=[]
    class_accuracy_across_slopes={}
    class_prediction_frequency={}

    for slope in sorted(tested_slopes.keys()):
        slope_output_path=OUTPUTS_PATH+"/"+tested_slopes[slope]
        with open(f"{slope_output_path}/model_accuracies","rb") as file:
            model_accuracies=pickle.load(file)

        # Getting the overall accuracy of the model
        accuracies_across_slopes.append(sum([1 for img_data in model_accuracies.values() if img_data["predicted_label"]==img_data["actual_label"]])/len(model_accuracies.keys()))

        # observing the prediction confidences of the models and comparing this to the confidence in the actual class
        predicted_confidences_by_actual_label={}
        actual_confidences_by_actual_label={}
        predicted_label_by_actual_label={}

        for img_id in sorted(model_accuracies.keys()):
            img_data=model_accuracies[img_id]
            actual_label=img_data["actual_label"]
            
            predicted_confidences_by_actual_label[actual_label]=predicted_confidences_by_actual_label.get(actual_label,[])+[img_data["predicted_confidence"]]
            actual_confidences_by_actual_label[actual_label]=actual_confidences_by_actual_label.get(actual_label,[])+[img_data["actual_confidence"]]
            predicted_label_by_actual_label[actual_label]=predicted_label_by_actual_label.get(actual_label,[])+[img_data["predicted_label"]]

        fig,ax=plt.subplots(nrows=1,ncols=1)
        ax.set_ylabel("model confidence")
        ax.hist(sum([v for v in predicted_confidences_by_actual_label.values()],[]),color="red",bins=100,alpha=0.5,label="predicted")
        ax.hist(sum([v for v in actual_confidences_by_actual_label.values()],[]),color="blue",bins=100,alpha=0.5,label="actual")
        plt.legend()
        plt.savefig(f"{slope_output_path}/confidences.png",bbox_inches="tight")
        plt.close()

        # observing the accuracies of the class across the slopes
        class_accuracies_within_slope={}

        for img_id,img_data in model_accuracies.items():
            actual_label=img_data["actual_label"]
            class_accuracies_within_slope[actual_label]=class_accuracies_within_slope.get(actual_label,[])+[float(img_data["predicted_label"]==actual_label)]
        
        for label,accs in class_accuracies_within_slope.items():
            class_accuracy_across_slopes[label]=class_accuracy_across_slopes.get(label,[])+[sum(accs)/len(accs)]

        # getting the frequency with which each class is predicted
        class_prediction_frequency[slope]={}
        for img_data in model_accuracies.values():
            class_prediction_frequency[slope][img_data["predicted_label"]]=class_prediction_frequency[slope].get(img_data["predicted_label"],0)+1
    
    # saving the accuracies to then use to plot the accuracies off all the interventions together
    if os.path.exists(f"inceptionv1/intervention/accuracies"):
        with open(f"inceptionv1/intervention/accuracies","rb") as file:
            accuracies=pickle.load(file)
    else:
        accuracies={}
    accuracies[intervention_layer]=accuracies_across_slopes
    with open(f"inceptionv1/intervention/accuracies","wb") as file:
        pickle.dump(accuracies,file)
    # plotting the accuracies across slopes for this intervention
    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlabel("slope")
    ax.set_ylabel("accuracy")
    ax.plot(sorted(tested_slopes.keys()),accuracies_across_slopes,color="blue")
    plt.savefig(f"{OUTPUTS_PATH}/accuracies.png",bbox_inches="tight")
    plt.close()

    # the line of the accuracy of each class against slope has an area and a variation. we calculate both of these
    # and imagine them on a scatter plots
    class_accuracy_area=[]
    class_accuracy_variation=[]
    for label in sorted(class_accuracy_across_slopes.keys()):
        accs=class_accuracy_across_slopes[label]
        area=line_area(sorted(tested_slopes.keys()),accs)
        variation=line_variation(sorted(tested_slopes.keys()),accs)
        if area!=0 and variation!=0:
            class_accuracy_area.append(area)
            class_accuracy_variation.append(variation)
    class_accuracy_area=np.array(class_accuracy_area)
    class_accuracy_variation=np.array(class_accuracy_variation)
    # we fit a line of best fit to this to get an overall trend which we then save
    # so we can plot the trends for each intervention together
    ppot,pcov=curve_fit(power_law,class_accuracy_area,class_accuracy_variation,p0=[1,-1])
    with open(f"{OUTPUTS_PATH}/fitted_parameters","wb") as file:
        pickle.dump({"a":ppot[0],"b":ppot[1]},file)

    x_fit=np.linspace(0.01,1,100)
    y_fit=power_law(x_fit,ppot[0],ppot[1])

    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlabel("area")
    ax.set_ylabel("variation")
    ax.scatter(class_accuracy_area,class_accuracy_variation,c="blue",s=2)
    ax.plot(x_fit,y_fit,color="orange")
    plt.savefig(f"{OUTPUTS_PATH}/class_accuracies_variation_against_area.png")
    plt.close()

    # plot a stacked bar chat of the frequency with which each class is predicted
    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 4096)
    num_distinct=[]
    for k in range(len(class_prediction_frequency.keys())-1):
        slope=sorted(class_prediction_frequency.keys())[k]
        next_slope=sorted(class_prediction_frequency.keys())[k+1]
        labels=sorted(class_prediction_frequency[slope].keys())
        frequencies=[class_prediction_frequency[slope][label] for label in labels]
        r=Rectangle((slope,0),next_slope-slope,frequencies[0],color=plt.cm.winter(labels[0]/1000))
        ax.add_patch(r)

        for j in range(1,len(labels)):
            r=Rectangle((slope,sum(frequencies[:j])),next_slope-slope,frequencies[j],color=plt.cm.winter(labels[j]/1000))
            ax.add_patch(r)
        num_distinct.append(len(labels))
    slope=sorted(class_prediction_frequency.keys())[-1]
    next_slope=slope+0.1
    labels=sorted(class_prediction_frequency[slope].keys())
    frequencies=[class_prediction_frequency[slope][label] for label in labels]
    r=Rectangle((slope,0),next_slope-slope,frequencies[0],color=plt.cm.winter(labels[0]/1000))
    ax.add_patch(r)

    for j in range(1,len(labels)):
        r=Rectangle((slope,sum(frequencies[:j])),next_slope-slope,frequencies[j],color=plt.cm.winter(labels[j]/1000))
        ax.add_patch(r)
    num_distinct.append(len(labels))
    num_distinct.append(len(labels))
    # we plot the number of distinct class that are predicted at each layer 
    # and we plot the capacity of each layer by noting how many filters it has
    ax.plot(sorted(tested_slopes.keys())+[sorted(tested_slopes.keys())[-1]+0.1],num_distinct,"red")
    ax.plot(sorted(tested_slopes.keys())+[sorted(tested_slopes.keys())[-1]+0.1],capacities[intervention_layer]*np.ones(len(num_distinct)),linestyle="dashed",color="black")
    plt.savefig(f"{OUTPUTS_PATH}/class_prediction_frequency.png")
    plt.close()

    # we now investigate the predictions when the layer is completely linear
    # and how predictions relate to the actual class
    slope=sorted(tested_slopes.keys())[-1]
    with open(f"{OUTPUTS_PATH}/{tested_slopes[slope]}/model_accuracies","rb") as file:
        model_accuracies=pickle.load(file)
    linear_predictions={}
    for img_data in model_accuracies.values():
        if img_data["predicted_class"] in linear_predictions.keys():
            linear_predictions[img_data["predicted_class"]][img_data["actual_class"]]=linear_predictions[img_data["predicted_class"]].get(img_data["actual_class"],0)+1
        else:
            linear_predictions[img_data["predicted_class"]]={}
            linear_predictions[img_data["predicted_class"]][img_data["actual_class"]]=1
    for predicted_class,actual_classes in linear_predictions.items():
        linear_predictions[predicted_class]["total"]=sum(actual_classes.values())

    with open(f"{OUTPUTS_PATH}/linear_prediction_frequency.txt","w") as file:
        for d in sorted(linear_predictions.items(),key=lambda item:item[1]["total"],reverse=True):
            file.write(f"{d[0]} - {d[1]["total"]}\n")
            for d2 in sorted(d[1].items(),key=lambda item:item[1],reverse=True):
                if d2[0]=="total":
                    continue
                file.write(f"    {d2[0]} - {d2[1]}\n")
    # we do the same thing now but switch the predicted and actual classes and summarise
    # the frequencies as a pie chart
    linear_predictions={}
    for img_data in model_accuracies.values():
        if img_data["actual_class"] in linear_predictions.keys():
            linear_predictions[img_data["actual_class"]][img_data["predicted_class"]]=linear_predictions[img_data["actual_class"]].get(img_data["predicted_class"],0)+1
        else:
            linear_predictions[img_data["actual_class"]]={}
            linear_predictions[img_data["actual_class"]][img_data["predicted_class"]]=1
    for actual_class,predicted_classes in linear_predictions.items():
        linear_predictions[actual_class]["distinct_predictions"]=len(predicted_classes.keys())

    with open(f"{OUTPUTS_PATH}/linear_actual_classes_predicted_frequency.txt","w") as file:
        for d in sorted(linear_predictions.items(),key=lambda item:item[1]["distinct_predictions"],reverse=True):
            file.write(f"{d[0]} - {d[1]["distinct_predictions"]}\n")
            for d2 in sorted(d[1].items(),key=lambda item:item[1],reverse=True):
                if d2[0]=="distinct_predictions":
                    continue
                file.write(f"    {d2[0]} - {d2[1]}\n")
    counts={}
    for actual_class,predicted_classes in linear_predictions.items():
        counts[predicted_classes["distinct_predictions"]]=counts.get(predicted_classes["distinct_predictions"],0)+1
    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.pie(x=counts.values(),labels=counts.keys(),colors=[plt.cm.brg(k/len(counts.keys())) for k in range(len(counts.keys()))])
    plt.savefig(f"{OUTPUTS_PATH}/linear_actual_classes_frequency.png",bbox_inches="tight")
    plt.close()


# layer_specific("inception3a")
# layer_specific("inception3b")
# layer_specific("inception4a")
# layer_specific("inception4b")
# layer_specific("inception4c")
# layer_specific("inception4d")
# layer_specific("inception4e")
# layer_specific("inception5a")
# layer_specific("inception5b")

def general():
    PATH="inceptionv1/intervention"

    with open("inceptionv1/intervention/accuracies","rb") as file:
        accuracies=pickle.load(file)
    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlabel("slope")
    ax.set_ylabel("accuracy")
    for (k,layer) in enumerate(sorted(accuracies.keys())):
        tested_slopes=[float(dir[0]+"."+dir[1:]) for dir in os.listdir(f"{PATH}/{layer}") if "0" in dir]
        ax.plot(tested_slopes,accuracies[layer],color=plt.cm.brg(k/len(accuracies.keys())),label=layer)
    plt.legend()
    plt.savefig(f"{PATH}/accuracies.png")

    layers=sorted([layer for layer in os.listdir(PATH) if "inception" in layer])

    fig,ax=plt.subplots(nrows=1,ncols=1)
    ax.set_xlabel("area")
    ax.set_ylabel("variation")
    x_fit=np.linspace(0.01,1,100)
    for (k,layer) in enumerate(layers):
        with open(f"{PATH}/{layer}/fitted_parameters","rb") as file:
            fitted_parameters=pickle.load(file)
        ax.plot(x_fit,power_law(x_fit,**fitted_parameters),c=plt.cm.brg(k/len(layers)),label=layer)
    plt.legend()
    plt.savefig(f"{PATH}/class_accuracies_variation_against_area.png")
    plt.close()

general()