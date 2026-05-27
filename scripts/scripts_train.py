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
from sklearn.model_selection import LeaveOneGroupOut

def get_annotation(annotation_file, n, list_holds, time):
    # Load annotation file
    if annotation_file.endswith('.xlsx'):
        annotation = pd.read_excel(annotation_file)
    elif annotation_file.endswith('.csv'):
        annotation = pd.read_csv(annotation_file)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx.")

    # Convert time to seconds if in MM:SS format
    time_annotation = annotation["Time"]
    if isinstance(time_annotation[0], str) and ':' in time_annotation[0]:
        time_annotation = [int(t.split(":")[0])*60 + float(t.split(":")[1]) for t in time_annotation]

    # Initialize output structure
    limbs = ['LH', 'RH', 'LF', 'RF']
    annotation_ts = {limb: {hold: np.zeros(n) for hold in list_holds} for limb in limbs}
    hold_states = {limb: {hold: 0 for hold in list_holds} for limb in limbs}

    # Process each annotation
    for _, row in annotation.iterrows():
        limb = row["Member parts"]
        hold_id = str(row["Holds ID"])
        action = row["Action"]

        if limb in limbs and hold_id in list_holds and hold_id != "-1":
            i = np.argmin(np.abs(time - time_annotation[_]))
            if action == "A" and hold_states[limb][hold_id] == 0:
                annotation_ts[limb][hold_id][i] = 1
                hold_states[limb][hold_id] = 1
            elif action == "D" and hold_states[limb][hold_id] == 1:
                annotation_ts[limb][hold_id][i] = -1
                hold_states[limb][hold_id] = 0

    # Ensure all holds end in a "not occupied" state
    for limb in annotation_ts:
        for hold in annotation_ts[limb]:
            if hold_states[limb][hold] == 1:
                annotation_ts[limb][hold][-1] = -1

    # convertit les A/D en binaire 
    for limb in annotation_ts:
        for hold in annotation_ts[limb]:
            annotation_ts[limb][hold] = np.cumsum(annotation_ts[limb][hold])

    return annotation_ts

def get_pos(tracking_file):
    pos = {}
    with open(tracking_file,'rb') as o:
        list_pos = pickle.load(o)
    for tp in ["RH","LH","RF","LF","H"]:
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
            h = f.split('_')[1].replace(".png","")
            ret, thresh = cv2.threshold(cv2.cvtColor(cv2.imread(os.path.join(folder_holds,f)), cv2.COLOR_BGR2GRAY), 127, 1, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            dist = cv2.distanceTransform(1-thresh, cv2.DIST_L2, 0)
            mask_holds[h] = dist.copy()
    return mask_holds
    
def get_dataset(sessions, dataset_file):

    x = []
    y = []
    m = []
    i = []
    for idx, s in enumerate(sessions):
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
            for l in mask_holds:
                dist = mask_holds[l][(p[:,1]/dy).astype(int),(p[:,0]/dx).astype(int)][:,np.newaxis]               
                x.append(np.concatenate((dist.T,velo.T)).T)
                y.append(annot[tp][l])   
                m.append(np.full(n, f"{tp} - Prise {l}")) # np.full pour être de même taille que annot[tp][l] et np.concatenate((dist.T,velo.T)).T
                i.append(np.full(n,idx))
    x = np.concatenate(x)
    y = np.concatenate(y)
    m = np.concatenate(m)
    i = np.concatenate(i)
    os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
    with open(dataset_file, 'wb') as handle:
        pickle.dump({'x':x,'y':y,'m': m, 'i' : i}, handle)
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
            for h in mask_holds:
                a.append(annotation[l][h])

                d.append(detection[l][h][:,0])
        f1.append(f1_score(np.concatenate(a),np.concatenate(d)))

    return np.mean(f1)

def get_f1_score2(u_net_file, dataset_file):
    u_net = torch.jit.load(u_net_file, map_location=device)
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
    

def train_NN(dataset_file, network_file, layers=[64,16],display=True,epochs=4500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(dataset_file,'rb') as o:
        dataset = pickle.load(o)
    
    # sous-échantillonnage + retrait des -1
    pos_idx = np.where(dataset["y"] == 1)[0]
    zero_idx = np.where(dataset["y"] == 0)[0]
    zero_idx_sampled = np.random.choice(zero_idx, size=len(pos_idx)*10, replace=False)
    idx = np.concatenate([pos_idx, zero_idx_sampled])
    np.random.shuffle(idx)
    
    X = torch.tensor(dataset["x"][idx]).float().to(device)
    Y = torch.tensor(dataset["y"][idx, None]).float().to(device)

    u_net = u_neural(X,Y,layers).to(device)
    list_loss = []
    optimizer = torch.optim.Adam(u_net.parameters(), lr=1e-4)
    bar_train = st.progress(0., text="Training progress")
    
    for epoch in range(epochs):
        bar_train.progress(epoch/epochs, text="Training progress")
        optimizer.zero_grad()
        loss = u_net.f1_loss()
        list_loss.append(loss.item())
        loss.backward()
        optimizer.step()

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

def train_kfold_NN(dataset_file, layers=[64,16], epochs=4000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(dataset_file, 'rb') as o:
        dataset = pickle.load(o)
    
    X = dataset["x"]
    Y = dataset["y"]
    I = dataset["i"]

    logo = LeaveOneGroupOut()
    scores_f1 = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(X, Y, I)):
        vid = np.unique(I[val_idx])
        print(f"fold : {fold + 1}\n")

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_val = torch.tensor(X[val_idx]).float().to(device)
        Y_val = torch.tensor(Y[val_idx]).float().to(device)
        
        pos_idx = np.where(Y_train == 1)[0]
        zero_idx = np.where(Y_train == 0)[0]
        zero_idx_sampled = np.random.choice(zero_idx, size=len(pos_idx)*10, replace=False)
        idx_sub = np.concatenate([pos_idx, zero_idx_sampled])
        np.random.shuffle(idx_sub)
        
        X_train = torch.tensor(X_train[idx_sub]).float().to(device)
        Y_train = torch.tensor(Y_train[idx_sub, None]).float().to(device)
        
        u_net = u_neural(X_train, Y_train, layers).to(device)
        optimizer = torch.optim.Adam(u_net.parameters(), lr=1e-4)

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = u_net.f1_loss()
            loss.backward()
            optimizer.step()

        u_net.eval()
        
        with torch.no_grad():
            Y_pred_val = u_net(X_val)
            
            Y_pred_binary = (Y_pred_val.cpu().numpy() > 0.5).astype(int).flatten()
            Y_true_binary = Y_val.cpu().numpy().flatten()
            
            Y_pred_clean = remove_short_sequences(Y_pred_binary, 20)
            
            y_true = Y_true_binary * (Y_true_binary > 0)
            y_pred = Y_pred_clean * (Y_pred_clean > 0)

            tp = (y_true * y_pred).sum()
            tn = ((1 - y_true) * (1 - y_pred)).sum()
            fp = ((1 - y_true) * y_pred).sum()
            fn = (y_true * (1 - y_pred)).sum()
            
            epsilon = 1e-7
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            
            fold_f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            
            scores_f1.append(fold_f1)
        print(f"Fold {fold + 1} terminé. F1-Score : {fold_f1:.4f}")

    print(f"Scores par fold : {scores_f1}")
    print(f"Score F1 Moyen  : {np.mean(scores_f1):.4f}")
        
    return scores_f1

def get_detection(sessions, dataset_out, u_net_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = []
    y = []
    y_score_pred = []
    y_score_true = []

    color = {'LH':'red','RH':'orange','LF':'blue','RF':'purple'}
    u_net = torch.jit.load(u_net_file, map_location=device)
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

            # bloc pour empecher d'avoir un membre sur 2 prises en même temps 
            proba = np.zeros((n, len(mask_holds)))
            temp = {} 
            
            for en, l in enumerate(mask_holds):
                dist = mask_holds[l][(p[:,1]/dy).astype(int),(p[:,0]/dx).astype(int)][:,np.newaxis]
                X = torch.tensor(np.concatenate((dist.T,velo.T)).T).float().to(device)
                Yout = u_net(X)
                proba[:, en] = Yout.detach().cpu().numpy().flatten()
                temp[l] = np.concatenate((dist.T,velo.T)).T
            
            best_i = np.argmax(proba, axis=1)
            max_proba = np.max(proba, axis=1)
            best_matrice = np.zeros_like(proba)

            for t in range(n):
                if max_proba[t] > 0.5:
                    best_matrice[t, best_i[t]] = 1

            for en, l in enumerate(mask_holds):
                y0[tp][l] = remove_short_sequences(best_matrice[:, en].astype(int), 20)
                x.append(temp[l])
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
                
                tp = (y_true * y_pred).sum()
                tn = ((1 - y_true) * (1 - y_pred)).sum()
                fp = ((1 - y_true) * y_pred).sum()
                fn = (y_true * (1 - y_pred)).sum()
                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                f1 = 2* (precision*recall) / (precision + recall + epsilon)
                plt.title(f"Final F1-score: {f1:.2f}")
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
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
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
    for l in list_holds:
        for p,tp in zip([pos_RH,pos_LH,pos_RF,pos_LF],["RH","LH","RF","LF"]):
            dist = np.linalg.norm(p-np.ones((p.shape[0],2))*np.array(l[1:]),axis=1)
            d = detect_low_values(dist,tresh_reach0,tresh_leave0)
            detection[tp][l] = 1-remove_short_true_sequences(1-remove_short_true_sequences(d,minimal_time0),minimal_time0)
    return detection
