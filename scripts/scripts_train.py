import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import sys
import re
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import Tensor
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def get_annotation_old(annotation_file,n,list_holds,time):
    annotation = pd.read_excel(annotation_file)
    annotation_ts = {'LH':{}, 'RH':{}, 'LF':{}, 'RF':{}}
    for l in annotation_ts:
        for en,h in enumerate(list_holds):
            annotation_ts[l][en] = np.zeros(n)

    for d in annotation.index:
        l = ''
        if annotation["Membre"][d] == 'MG':
            l = 'LH'
        if annotation["Membre"][d] == 'MD':
            l = 'RH'
        if annotation["Membre"][d] == 'PG':
            l = 'LF'
        if annotation["Membre"][d] == 'PD':
            l = 'RF'

        if l != '' and annotation["Prise ID"][d]>-1:
            i = np.argmin(np.abs(time-annotation["Temps"][d]))
            if annotation["AD"][d] == "A":
                annotation_ts[l][annotation["Prise ID"][d]][i] = 1
            if annotation["AD"][d] == "D":
                annotation_ts[l][annotation["Prise ID"][d]][i] = -1
    for l in annotation_ts:
        for en,h in enumerate(list_holds):
            annotation_ts[l][en] = np.cumsum(annotation_ts[l][en])
    return annotation_ts


def get_annotation(annotation_file,n,list_holds,time):
    if '.xlsx' in annotation_file:
        annotation = pd.read_excel(annotation_file)
    if '.csv' in annotation_file:
        annotation = pd.read_csv(annotation_file)
    time_annotation = annotation["Time"]
    if ':' in time_annotation[0]:
        time_annotation = [int(t.split(":")[0])*60+float(t.split(":")[1]) for t in time_annotation]
    #print('ta:',time_annotation[0])
    annotation_ts = {'LH':{}, 'RH':{}, 'LF':{}, 'RF':{}}
    for l in annotation_ts:
        for en,h in enumerate(list_holds):
            annotation_ts[l][h] = np.zeros(n) # changement 17/01

    for d in annotation.index:
        l = annotation["Member parts"][d]
        if l != '' and annotation["Holds ID"][d]>-1:
            #print('taa,d',annotation["Time"][d],d)
            i = np.argmin(np.abs(time-time_annotation[d]))
            if annotation["Action"][d] == "A" and annotation["Holds ID"][d] != 0:
#                print("d:",d,"l:",l,"hid:",annotation["Holds ID"][d])
#                for w in annotation_ts[l]:
#                    print("annot ts[l]:",w)
                annotation_ts[l][int(annotation["Holds ID"][d])][i] = 1
            if annotation["Action"][d] == "D" and annotation["Holds ID"][d] != 0:
                annotation_ts[l][int(annotation["Holds ID"][d])][i] = -1
    for l in annotation_ts:
        for en,h in enumerate(list_holds):
#            annotation_ts[l][en] = np.cumsum(annotation_ts[l][en]) # changement 16/01
            annotation_ts[l][h] = np.cumsum(annotation_ts[l][h])
    return annotation_ts

def get_pos(tracking_file):
    pos = {}
    with open(tracking_file,'rb') as o:
        list_pos = pickle.load(o)
    for tp in ["RH","LH","RF","LF","H"]:
#        detection[i0][tp] = {}
        time = list_pos["Time (s)"]
        n = time.shape[0]
        dt = np.median(np.diff(time))

        p = np.zeros((n,2))
        p[:,0] = list_pos[tp+"x (pix)"]
        p[:,1] = list_pos[tp+"y (pix)"]
        pos[tp] = p.copy()
    pos['time'] = time.copy()
    return pos, list_pos["dx"], list_pos["dy"]
    

def get_mask_holds(folder_holds):
    mask_holds = {}
    for f in os.listdir(folder_holds):  
        if "hold_" in f:
            h = int(f.split('_')[1].replace(".png",""))
            ret, thresh = cv2.threshold(cv2.cvtColor(cv2.imread(os.path.join(folder_holds,f)), cv2.COLOR_BGR2GRAY), 127, 1, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            dist = cv2.distanceTransform(1-thresh, cv2.DIST_L2, 0)
            mask_holds[h] = dist.copy()
    return mask_holds
    
def get_dataset(sessions, dataset_file):

    x = []
    y = []
    for s in sessions:
        p = s["position"]
        a = s["annotation"]
        h = s["mask"]
        list_pos, dx, dy = get_pos(p.strip())
        time = list_pos["time"]
        dt = np.median(np.diff(time))
        mask_holds = get_mask_holds(h.strip())
        n = time.shape[0]
        annot = get_annotation(a.strip(),n, mask_holds, time)
        for tp in ["RH","LH","RF","LF"]:
            p = list_pos[tp]     
            p[:,0] = p[:,0]*dx
            p[:,1] = p[:,1]*dy
            velo = np.diff(p,axis=0)/dt
            velo = np.concatenate((velo[0:1,:],velo))
            acc = np.diff(velo,axis=0)/dt
            acc = np.concatenate((acc[0:1,:],acc))
            for en,l in enumerate(mask_holds):
                dist = mask_holds[l][(p[:,1]/dx).astype(int),(p[:,0]/dy).astype(int)][:,np.newaxis]
#                    dist = mask_holds[en][(p[:,1]/dy).astype(int),(p[:,0]/dx).astype(int)][:,np.newaxis] # changement 16/01
#                except IndexError:
#                    print("en:",en,"l:",l,"mask_size:",mask_holds[l].shape, "p1m:", np.max((p[:,1]/dy).astype(int)), "p0m:", np.max((p[:,0]/dx).astype(int)),"p1init:", (p[0,1]/dy).astype(int), "p0init:", (p[0,0]/dx).astype(int) )
#                finally:
#                    return                    
                x.append(np.concatenate((dist.T,velo.T)).T)
#                    y.append(annot[tp][en])   # changement 16/01
                y.append(annot[tp][l])   
    x = np.concatenate(x)
    y = np.concatenate(y)
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    with open(dataset_file, 'wb') as handle:
        pickle.dump({'x':x,'y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
    


   
class u_neural(nn.Module):
    def __init__(self,X,Y,layers):
        super().__init__()
        self.X = X
        self.Y = Y
        mod = []
        if X is None:
            mod.append(nn.Linear(3,layers[0]))
        else:
            mod.append(nn.Linear(X.shape[1],layers[0]))
        mod.append(nn.Tanh())
        for i in range(len(layers)-1):
            mod.append(nn.Linear(layers[i],layers[i+1]))
            mod.append(nn.Tanh())
        mod.append(nn.Linear(layers[-1],1))
        mod.append(nn.Sigmoid())
        self.net = nn.Sequential(*mod)
        return
    
    def forward(self, x):
        return self.net(x)

    def loss(self):
        return torch.nn.functional.binary_cross_entropy(self(self.X),self.Y)
        return torch.nn.functional.mse_loss(self(self.X),self.Y)
    
    def f1_loss(self):
        y_true = self.Y
        y_pred = self(self.X)
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2* (precision*recall) / (precision + recall + epsilon)
        return -f1

def get_f1_score(u_net,list_file_pos,list_file_annotation,list_folder_holds):
    list_pos = {}
    for f in list_file_pos:
        list_pos[f.split("/")[-2]] = get_pos(f)
    f1 = []
    for i,i0 in enumerate(list_pos):
        a = []
        d = []
        annotation_file = list_file_annotation[i]
        mask_holds = get_mask_holds(list_folder_holds[i])
        n = list_pos[i0]["RH"].shape[0]
        time = list_pos[i0]["time"]
        annotation = get_annotation(annotation_file,n,mask_holds,time)
        detection = get_detection(list_pos[i0],mask_holds,u_net)
        for l in annotation:
            for en,h in enumerate(mask_holds):
                a.append(annotation[l][h])

                d.append(detection[l][h][:,0])
        f1.append(f1_score(np.concatenate(a),np.concatenate(d)))

    return np.mean(f1)

def get_f1_score2(u_net_file, dataset_file):
    u_net = torch.jit.load(u_net_file)
    u_net.eval()
    with open(dataset_file,'rb') as o:
        dataset = pickle.load(o)
    X = torch.tensor(dataset["x"]).float()
    Y = torch.tensor(dataset["y"][:,None]).float()
    Y = Y*(Y>0) #Just in case the script to read the annotation gives negative values
    y_pred = (u_net(X)>0.5).to(torch.float32)
    y_true = Y
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.detach().numpy().tolist()
    

def train_NN(dataset_file, network_file, layers=[10]*2,display=True,epochs=2000):
    with open(dataset_file,'rb') as o:
        dataset = pickle.load(o)
    X = torch.tensor(dataset["x"]).float()
    Y = torch.tensor(dataset["y"][:,None]).float()
    u_net = u_neural(X,Y,layers)
    list_loss = []
    optimizer = torch.optim.Adam(u_net.parameters(), lr=1e-3)
    bar_train = st.progress(0., text="Training progress")
    for epoch in range(epochs):
        bar_train.progress(epoch/epochs, text="Training progress")
        optimizer.zero_grad()
        loss = u_net.f1_loss()
        list_loss.append(loss.item())
        loss.backward()
        optimizer.step()
#    print("Final f1-score:",-list_loss[-1])
    if display:
        fig = plt.figure()
        plt.plot([-l for l in list_loss])
        plt.xlabel("Iterations")
        plt.ylabel("F1-score")
        plt.grid()
        #st.pyplot(fig)
    else:
        fig = None
    os.makedirs(os.path.dirname(network_file), exist_ok=True)
    model_scripted = torch.jit.script(u_net) # Export to TorchScript
    model_scripted.save(network_file) # Save
    return list_loss, fig



def get_detection(sessions, dataset_out, u_net_file):
    x = []
    y = []
    y_score_pred = []
    y_score_true = []

    color = {'LH':'red','RH':'orange','LF':'blue','RF':'purple'}
    u_net = torch.jit.load(u_net_file)
    u_net.eval()
    bar_inference = st.progress(0., text="Inference progress")
    for i,s in enumerate(sessions):
        bar_inference.progress(i/len(sessions), text="Inference progress")
        # Perform inference
        p = s["position"]
        det = s["prediction"]        
        h = s["mask"]
        list_pos, dx, dy = get_pos(p.strip())
        time = list_pos["time"]
        dt = np.median(np.diff(time))
        mask_holds = get_mask_holds(h.strip())
        n = time.shape[0]
        y0 = {}
        y0['time'] = time
        # If a plot is required
        if "plot" in s:
            fig = s["plot"]
        else:
            fig = None
        # If annotation is available
        if "annotation" in s:
            annot = get_annotation(s["annotation"].strip(),n, mask_holds, time)
            y_true = []
            y_pred = []
        else:
            annot = None

        if fig is not None:
            plt.figure(figsize=(10,len(mask_holds)))
            plt.tight_layout()

        detection_out = {"Time":time}
        for tp in ["RH","LH","RF","LF"]:
            detection_out[tp] = {}
            p = list_pos[tp]     
            p[:,0] = p[:,0]*dx
            p[:,1] = p[:,1]*dy
            velo = np.diff(p,axis=0)/dt
            velo = np.concatenate((velo[0:1,:],velo))
            acc = np.diff(velo,axis=0)/dt
            acc = np.concatenate((acc[0:1,:],acc))
            y0[tp] = {}
            for en,l in enumerate(mask_holds):
                dist = mask_holds[l][(p[:,1]/dx).astype(int),(p[:,0]/dy).astype(int)][:,np.newaxis]
#
#                print("l:",l,"mask_size:",mask_holds[l].shape, "p1m:", np.max((p[:,1]/dy).astype(int)), "p0m:", np.max((p[:,0]/dy).astype(int)),"p1init:", (p[0,1]/dy).astype(int), "p0init:", (p[0,0]/dy).astype(int) )
#                dist = mask_holds[en][(p[:,1]/dy).astype(int),(p[:,0]/dx).astype(int)][:,np.newaxis] # changement 16/01
#                try:
#                except IndexError:
#                   print("tp:",tp, "en:",en,"l:",l,"mask_size:",mask_holds[l].shape, "p1m:", np.max((p[:,1]/dy).astype(int)), "p0m:", np.max((p[:,0]/dx).astype(int)),"p1init:", (p[0,1]/dy).astype(int), "p0init:", (p[0,0]/dx).astype(int) )
#                finally:
#                    return 0
                X = torch.tensor(np.concatenate((dist.T,velo.T)).T).float()
                Yout = u_net(X)
#                print("Size Yout:", Yout.shape)
                y0[tp][l] = remove_short_sequences((Yout.detach().numpy()>0.5).astype(int),20)
                x.append(np.concatenate((dist.T,velo.T)).T)
                y.append(y0[tp][l].flatten())
                
                detection_out[tp][l] = y0[tp][l].flatten().copy()
                if fig is not None:
                    plt.subplot(len(mask_holds),1,en+1)
                    plt.plot(time,y0[tp][l], c=color[tp], linestyle='solid', label=tp)
                    if annot is not None:
                        plt.plot(time,annot[tp][l], c=color[tp], linestyle='dotted')
                    plt.ylim(0,1.1)
                    plt.ylabel("Hold:"+str(l))
                if annot is not None:
                    y_true.append(annot[tp][l].flatten())
                    y_pred.append(y0[tp][l].flatten())
                    y_score_true.append(annot[tp][l].flatten())
                    y_score_pred.append(y0[tp][l].flatten())

        
        if fig is not None:
            plt.subplot(len(mask_holds),1,1)
            plt.legend()
            # If there is annotation, compute the F1-score
            if annot is not None:
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)

                y_true = y_true * (y_true>0) # Mainly in case of an annotation issue
                y_pred = y_pred * (y_pred>0)
                
#               print("y_true shape",y_true.shape)
#               print("y_pred shape",y_pred.shape)
                
                tp = (y_true * y_pred).sum()
#                print("tp",tp)
                tn = ((1 - y_true) * (1 - y_pred)).sum()
#                print("tn",tn)
                fp = ((1 - y_true) * y_pred).sum()
#                print("fp",fp)
                fn = (y_true * (1 - y_pred)).sum()
 #               print("fn",fn)
                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2* (precision*recall) / (precision + recall + epsilon)
                plt.title(f"Final F1-score: {f1:.2f}")
#                print("Score is",f1)
            plt.subplot(len(mask_holds),1,len(mask_holds))
            plt.xlabel("Time (s)")

            os.makedirs(os.path.dirname(fig), exist_ok=True)
            plt.savefig(fig)
            # Here, just to plot the aligned sequence, just used for debugging
#            plt.figure()
#            plt.plot(y_true)
#            plt.plot(y_pred)
#            plt.savefig(fig.replace('.png','_.png'))


    bar_inference.progress(1., text="Inference progress")

    x = np.concatenate(x)
    y = np.concatenate(y)

    os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
    with open(dataset_out, 'wb') as handle:
        pickle.dump({'x':x,'y':y}, handle)
    with open(det, 'wb') as handle:
        pickle.dump(detection_out, handle)

    if annot is None:
        return 1
    y_true = np.concatenate(y_score_true)
    y_pred = np.concatenate(y_score_pred)

    y_true = y_true * (y_true>0) # Mainly in case of an annotation issue
    y_pred = y_pred * (y_pred>0)
                
    
    tp = (y_true * y_pred).sum()
#    print("tp",tp)
    tn = ((1 - y_true) * (1 - y_pred)).sum()
#    print("tn",tn)
    fp = ((1 - y_true) * y_pred).sum()
#    print("fp",fp)
    fn = (y_true * (1 - y_pred)).sum()
#    print("fn",fn)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    
    return f1


def detect_low_values(signal, low_threshold, high_threshold):
    """
    Detect low values with hysteresis in a 1D signal.
    
    Args:
    - signal: The input 1D numpy array.
    - low_threshold: The lower threshold value.
    - high_threshold: The higher threshold value.
    
    Returns:
    - A binary array where values less than the low_threshold are set to 1,
      values between low_threshold and high_threshold are set to 0 or 1 based on
      connectivity, and values greater than the high_threshold are set to 0.
    """
    # Initialize output array
    output = np.zeros_like(signal, dtype=np.int8)
    
    # Apply low threshold
    output[signal < low_threshold] = 1
    
    # Find indices of values between low and high thresholds
    between_indices = np.where((signal >= low_threshold) & (signal <= high_threshold))[0]
    
    # Perform hysteresis thresholding
    for index in between_indices:
        if np.any(output[max(0, index-1):min(len(signal), index+2)]):
            output[index] = 1
    
    return output

def remove_short_true_sequences(boolean_array, min_sequence_length):
    """
    Remove short sequences of True values from a 1D array of booleans.

    Args:
    - boolean_array: 1D NumPy array of boolean values.
    - min_sequence_length: Minimum length of the True sequence to be kept.

    Returns:
    - A new 1D array with short True sequences removed.
    """
    result = boolean_array.copy()
    in_sequence = 0
    start_index = 0

    for i, value in enumerate(boolean_array):
        if value:
            if not in_sequence:
                start_index = i
                in_sequence = 1
        elif in_sequence:
            sequence_length = i - start_index
            if sequence_length < min_sequence_length:
                result[start_index:i] = 0
            in_sequence = 0

    # Check if the last sequence extends to the end of the array
    if in_sequence:
        sequence_length = len(boolean_array) - start_index
        if sequence_length < min_sequence_length:
            result[start_index:] = 0

    return result

def remove_short_sequences(arr,min_length):
    return 1-remove_short_true_sequences(1-remove_short_true_sequences(arr,min_length),min_length)
        
def realize_detection(tresh_reach0,tresh_leave0,minimal_time0,lambda_velocity):
    global list_holds, pos_RH,pos_LH,pos_RF,pos_LF
    detection = {"RH":{},"LH":{},"RF":{},"LF":{}}
    for en,l in enumerate(list_holds):
        for p,tp in zip([pos_RH,pos_LH,pos_RF,pos_LF],["RH","LH","RF","LF"]):
            dist = np.linalg.norm(p-np.ones((p.shape[0],2))*np.array(l[1:]),axis=1)
            d = detect_low_values(dist,tresh_reach0,tresh_leave0)
            detection[tp][l] = 1-remove_short_true_sequences(1-remove_short_true_sequences(d,minimal_time0),minimal_time0)
    return detection
