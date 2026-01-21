import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sys
import re
from scipy import interpolate
from scipy.signal import butter, filtfilt
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
import streamlit as st
import os
import subprocess

'''
def yolo_tracking(video_file, tracking_yolo, title= None, out_image_name=None, model_name = 'yolov8l.pt', display = True, stream=True):
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened() == False:
        st.write("Error opening video stream or file")
        raise TypeError
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    st.write("    Video frame size:",frame_width,frame_height)
    st.write("    Video duration:",length,"frames")
    model = YOLO(model_name)  # Load an official Detect model
    st.write(f"Device to run the model: {model.device}") 
    results = model.track(source=video_file, classes=[0], show=False,verbose=False,stream=stream)
    list_detection = []
    bar_yolo = st.progress(0., text="Tracking progress")
 #   for i, r in enumerate(stqdm.stqdm(results, total=length, desc="Tracking progress", key="yolo")):
    for i, r in enumerate(results):#, total=length, desc="Tracking progress", key="yolo")):
        bar_yolo.progress(i/length, text="Tracking progress")

        try:
            rb  = r.boxes.xyxy.cpu().numpy()
            rid = r.boxes.id.int().cpu().tolist()
            rt  = (i/fps)*1000
            kp  = r.keypoints.cpu().numpy()
            list_detection.append({'timestamp':rt,'boxes':rb,'id':rid, 'keypoints':kp})
        except:
            pass
    bar_yolo.empty()

    l3 = []
    for l in list_detection:
        for l2 in l['id']:
            l3.append(l2)
    if display:
        fig, ax = plt.subplots()
        m = 0
        tracking_person2 = -1
        for idx in set(l3):
            tracking_person = idx
            lx = []
            ly = []
            lt = []
            ii = 0
            for l in list_detection:
                ii += 1
                b = l['boxes']
                i = l['id']
                t = l['timestamp']

                if tracking_person in i:
                    id_box = i.index(tracking_person)
                    lx.append((b[id_box,0]+b[id_box,2])/2)
                    ly.append(frame_height-(b[id_box,1]+b[id_box,3])/2)
                    lt.append(t)

            if len(lx)>100: # Only consider detections longer than 100 frames
                ax.plot(lx,ly,label=str(idx))
                if max(ly)>m:
                    m = max(ly)
                    tracking_person2 = tracking_person
        ax.legend()
        st.pyplot(fig)
        os.makedirs(os.path.dirname(out_image_name), exist_ok=True)
        plt.savefig(out_image_name)
        print('Will be tracking person n°',tracking_person2)
    else:
        myfig = plt.figure()
        m = 0
        tracking_person2 = -1
        for idx in set(l3):
            tracking_person = idx
            lx = []
            ly = []
            lt = []
            ii = 0
            for l in list_detection:
                ii += 1
                b = l['boxes']
                i = l['id']
                t = l['timestamp']

                if tracking_person in i:
                    id_box = i.index(tracking_person)
                    lx.append((b[id_box,0]+b[id_box,2])/2)
                    ly.append(frame_height-(b[id_box,1]+b[id_box,3])/2)
                    lt.append(t)

            if len(lx)>100: # Only consider detections longer than 100 frames
                plt.plot(lx,ly,label=str(idx))
                if max(ly)>m:
                    m = max(ly)
                    tracking_person2 = tracking_person
        plt.legend()
        plt.title(title+" tracking climber:"+str(tracking_person2))
        os.makedirs(os.path.dirname(out_image_name), exist_ok=True)
        plt.savefig(out_image_name)
        plt.close(myfig)
    os.makedirs(os.path.dirname(tracking_yolo), exist_ok=True)
    with open(tracking_yolo, 'wb') as handle:
            pickle.dump({'list_detection': list_detection, 'tracking_person':[tracking_person2]}, handle)
    return list_detection, [tracking_person2]
    
    '''




def yolo_tracking(video_file, tracking_yolo, out_image_name=None, heatmap_image_name=None, model_name='yolov8l.pt', title=None, display=True, stream=True):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.write("Error opening video stream or file")
        raise TypeError
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    st.write("Video frame size:", frame_width, frame_height)
    st.write("Video duration:", length, "frames")

    model = YOLO(model_name)
    st.write(f"Device to run the model: {model.device}") 

    results = model.track(source=video_file, classes=[0], show=False, verbose=False, stream=stream)
    list_detection = []
    bar_yolo = st.progress(0., text="Tracking progress")

    # --- MATRICE TAMPON POUR LA HEATMAP ---
    heatmap_acc = np.zeros((frame_height, frame_width), dtype=np.float32)

    for i, r in enumerate(results):
        bar_yolo.progress(i / length, text="Tracking progress")

        try:
            rb = r.boxes.xyxy.cpu().numpy()
            rid = r.boxes.id.int().cpu().tolist()
            rt = (i / fps) * 1000
            kp = r.keypoints.cpu().numpy()
            list_detection.append({'timestamp': rt, 'boxes': rb, 'id': rid, 'keypoints': kp})

            # --- INCREMENTATION DU TAMPON ---
            for j, id_val in enumerate(rid):
                x_center = int((rb[j, 0] + rb[j, 2]) / 2)
                y_center = int((rb[j, 1] + rb[j, 3]) / 2)
                if 0 <= x_center < frame_width and 0 <= y_center < frame_height:
                    heatmap_acc[y_center, x_center] += 1  # incrémente de 1

        except:
            pass
    bar_yolo.empty()

    # --- TRACKING EXISTANT PAR PERSONNE ---
    l3 = []
    for l in list_detection:
        for l2 in l['id']:
            l3.append(l2)

    if display:
        fig, ax = plt.subplots()
        m = 0
        tracking_person2 = -1
        for idx in set(l3):
            tracking_person = idx
            lx, ly, lt = [], [], []
            for l in list_detection:
                b = l['boxes']
                i = l['id']
                t = l['timestamp']
                if tracking_person in i:
                    id_box = i.index(tracking_person)
                    lx.append((b[id_box,0]+b[id_box,2])/2)
                    ly.append(frame_height-(b[id_box,1]+b[id_box,3])/2)
                    lt.append(t)
            if len(lx) > 100:
                ax.plot(lx, ly, label=str(idx))
                if max(ly) > m:
                    m = max(ly)
                    tracking_person2 = tracking_person
        ax.legend()
        st.pyplot(fig)
        if out_image_name:
            os.makedirs(os.path.dirname(out_image_name), exist_ok=True)
            plt.savefig(out_image_name)
    else:
        myfig = plt.figure()
        m = 0
        tracking_person2 = -1
        for idx in set(l3):
            tracking_person = idx
            lx, ly, lt = [], [], []
            for l in list_detection:
                b = l['boxes']
                i = l['id']
                t = l['timestamp']
                if tracking_person in i:
                    id_box = i.index(tracking_person)
                    lx.append((b[id_box,0]+b[id_box,2])/2)
                    ly.append(frame_height-(b[id_box,1]+b[id_box,3])/2)
                    lt.append(t)
            if len(lx) > 100:
                plt.plot(lx, ly, label=str(idx))
                if max(ly) > m:
                    m = max(ly)
                    tracking_person2 = tracking_person
        plt.legend()
        plt.title((title or "") + " tracking climber:" + str(tracking_person2))
        if out_image_name:
            os.makedirs(os.path.dirname(out_image_name), exist_ok=True)
            plt.savefig(out_image_name)
        plt.close(myfig)

    # --- CREATION DE LA HEATMAP ---
    if heatmap_image_name is not None:
        folder = os.path.dirname(heatmap_image_name)
        if folder != '':
            os.makedirs(folder, exist_ok=True)

        print("Heatmap min/max avant normalisation:", heatmap_acc.min(), heatmap_acc.max())

        # --- Lissage pour obtenir un effet "traînée"
        heatmap_blur = cv2.GaussianBlur(heatmap_acc, (25, 25), 0)  # ajuster le kernel pour la douceur

        # Normalisation 0-255
        heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Appliquer colormap (JET) pour bleu->rouge
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Pixels à 0 deviennent blanc
        mask_zero = heatmap_norm == 0
        heatmap_color[mask_zero] = [255, 255, 255]  # BGR

        # Sauvegarde
        cv2.imwrite(heatmap_image_name, heatmap_color)
        st.write(f"Heatmap saved at {heatmap_image_name}")

        # Affichage Streamlit
        if display:
            st.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), caption="Trail Heatmap", width=700)

    # --- SAUVEGARDE PICKLE ---
    os.makedirs(os.path.dirname(tracking_yolo), exist_ok=True)
    with open(tracking_yolo, 'wb') as handle:
        pickle.dump({'list_detection': list_detection, 'tracking_person':[tracking_person2]}, handle)

    return list_detection, [tracking_person2], heatmap_acc



    
def mediapipe_tracking(video_file, tracking_yolo, tracking_initial):
    with open(tracking_yolo,'rb') as o:
        track_yolo = pickle.load(o)
    list_detection = track_yolo['list_detection']
    list_tracking_person = track_yolo['tracking_person']
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.8) 

    list_pose= []
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened() == False:
        st.write("Error opening video stream or file")
        raise TypeError
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    t_idx = 0
    bar_mediapipe = st.progress(0., text="Tracking progress")
    while cap.isOpened():
        bar_mediapipe.progress(t_idx/total_frames, text="Tracking progress")
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if t_idx>len(list_detection)-1:
            break
        for tracking_person in list_tracking_person:
            if tracking_person in list_detection[t_idx]['id']:
                i_idx = list_detection[t_idx]['id'].index(tracking_person)
                b = list_detection[t_idx]['boxes'][i_idx,:]
                image2 = image[int(b[1]):int(b[3]),int(b[0]):int(b[2]),:]
                results = pose.process(image2)
                rp = results.pose_landmarks
                rpw = results.pose_world_landmarks
                if rp is not None:
        #            lrp = [{'x':l.x,'y':l.y, 'z':l.z, 'visib':l.visibility} for l in rp.landmark]
                    lrp = [{'x':b[0]+(b[2]-b[0])*l.x,'y':b[1]+(b[3]-b[1])*l.y, 'z':l.z, 'visib':l.visibility} for l in rp.landmark]
                    lrpw= [{'x':l.x,'y':l.y, 'z':l.z, 'visib':l.visibility} for l in rpw.landmark]
                    list_pose.append({'timestamp':list_detection[t_idx]['timestamp'],'pose':lrp, 'world pose':lrpw, 'size':(width, height)})
                break
        t_idx += 1

    pose.close()
    cap.release()

    save = True # Saving the tracking
    if save:
        os.makedirs(os.path.dirname(tracking_initial), exist_ok=True)
        with open(tracking_initial, 'wb') as handle:
            pickle.dump(list_pose, handle)
    bar_mediapipe.empty()
    return
    
def extract_tracking(tracking_initial, pickle_tracking_file, xlsx_tracking_file, w_c, dx, dy):
    with open(tracking_initial,'rb') as o:
        list_pose = pickle.load(o)
    #w_c = 0.1 # cutting Shannon pulsation

    width  = list_pose[0]['size'][0]
    height = list_pose[0]['size'][1]    

    time = np.array([l['timestamp'] for l in list_pose])
    dt = np.median(np.diff(time))
    time2=np.arange(time[0],time[-1]+dt,dt)
    time2[-1] = min(time2[-1],time[-1])
    b,a = butter(3,w_c,btype='low')

    pos  = np.array([[l['pose'][23]['x'],l['pose'][23]['y']] for l in list_pose])
    pos += np.array([[l['pose'][24]['x'],l['pose'][24]['y']] for l in list_pose])
    pos_H = filtfilt(b,a,pos/2,axis=0)
    f = interpolate.interp1d(time,pos_H,axis=0)
    pos_H = f(time2)

    pos  = np.array([[l['pose'][15]['x'],l['pose'][15]['y']] for l in list_pose])
    pos += np.array([[l['pose'][21]['x'],l['pose'][21]['y']] for l in list_pose])
    pos += np.array([[l['pose'][19]['x'],l['pose'][19]['y']] for l in list_pose])
    pos += np.array([[l['pose'][17]['x'],l['pose'][17]['y']] for l in list_pose])
    pos_LH = filtfilt(b,a,pos/4,axis=0)
    f = interpolate.interp1d(time,pos_LH,axis=0)
    pos_LH = f(time2)

    pos  = np.array([[l['pose'][16]['x'],l['pose'][16]['y']] for l in list_pose])
    pos += np.array([[l['pose'][22]['x'],l['pose'][22]['y']] for l in list_pose])
    pos += np.array([[l['pose'][20]['x'],l['pose'][20]['y']] for l in list_pose])
    pos += np.array([[l['pose'][18]['x'],l['pose'][18]['y']] for l in list_pose])
    pos_RH = filtfilt(b,a,pos/4,axis=0)
    f = interpolate.interp1d(time,pos_RH,axis=0)
    pos_RH = f(time2)

    pos  = np.array([[l['pose'][27]['x'],l['pose'][27]['y']] for l in list_pose])
    pos += np.array([[l['pose'][29]['x'],l['pose'][29]['y']] for l in list_pose])
    pos += np.array([[l['pose'][31]['x'],l['pose'][31]['y']] for l in list_pose])
    pos_LF = filtfilt(b,a,pos/3,axis=0)
    f = interpolate.interp1d(time,pos_LF,axis=0)
    pos_LF = f(time2)

    pos  = np.array([[l['pose'][28]['x'],l['pose'][28]['y']] for l in list_pose])
    pos += np.array([[l['pose'][30]['x'],l['pose'][30]['y']] for l in list_pose])
    pos += np.array([[l['pose'][32]['x'],l['pose'][32]['y']] for l in list_pose])
    pos_RF = filtfilt(b,a,pos/3,axis=0)
    f = interpolate.interp1d(time,pos_RF,axis=0)
    pos_RF = f(time2)

    data = {'Time (s)':time2/1000.}

    s = pos_LF
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['LFx (pix)'] = np.clip(s[:,0],0,width-1)
    data['LFy (pix)'] = np.clip(s[:,1],0,height-1)
    data['LFv (pix/s)'] = v
    s = pos_LF.copy()
    s[:,0] *= dx
    s[:,1] *= dy
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['LFx (m)'] = s[:,0]
    data['LFy (m)'] = s[:,1]
    data['LFv (m/s)'] = v

    s = pos_RF
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['RFx (pix)'] = np.clip(s[:,0],0,width-1)
    data['RFy (pix)'] = np.clip(s[:,1],0,height-1)
    data['RFv (pix/s)'] = v
    s = pos_RF.copy()
    s[:,0] *= dx
    s[:,1] *= dy
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['RFx (m)'] = s[:,0]
    data['RFy (m)'] = s[:,1]
    data['RFv (m/s)'] = v


    s = pos_LH
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['LHx (pix)'] = np.clip(s[:,0],0,width-1)
    data['LHy (pix)'] = np.clip(s[:,1],0,height-1)
    data['LHv (pix/s)'] = v
    s = pos_LH.copy()
    s[:,0] *= dx
    s[:,1] *= dy
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['LHx (m)'] = s[:,0]
    data['LHy (m)'] = s[:,1]
    data['LHv (m/s)'] = v

    s = pos_RH
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['RHx (pix)'] = np.clip(s[:,0],0,width-1)
    data['RHy (pix)'] = np.clip(s[:,1],0,height-1)
    data['RHv (pix/s)'] = v
    s = pos_RH.copy()
    s[:,0] *= dx
    s[:,1] *= dy
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['RHx (m)'] = s[:,0]
    data['RHy (m)'] = s[:,1]
    data['RHv (m/s)'] = v

    s = pos_H
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['Hx (pix)'] = np.clip(s[:,0],0,width-1)
    data['Hy (pix)'] = np.clip(s[:,1],0,height-1)
    data['Hv (pix/s)'] = v
    s = pos_H.copy()
    s[:,0] *= dx
    s[:,1] *= dy
    v = np.linalg.norm(np.diff(s,axis=0),axis=1)/dt
    v = np.insert(v,0,v[0])
    data['Hx (m)'] = s[:,0]
    data['Hy (m)'] = s[:,1]
    data['Hv (m/s)'] = v

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(xlsx_tracking_file), exist_ok=True)
    os.makedirs(os.path.dirname(pickle_tracking_file), exist_ok=True)

    df.to_excel(xlsx_tracking_file)
    with open(pickle_tracking_file, 'wb') as handle:
        data['dx'] = dx
        data['dy'] = dy
        pickle.dump(data, handle)
    return 
    
def extract_median_image(video_file, image_file):
    cap = cv2.VideoCapture(video_file)
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if frame is not None:
            frames.append(frame)
    cap.release()
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    os.makedirs(os.path.dirname(image_file), exist_ok=True)
    cv2.imwrite(image_file, median_frame) 
    return
    

def write_on_video(video_file, video_out, tracking_file):
    pos = {}
    with open(tracking_file,'rb') as o:
        list_pos = pickle.load(o)
    for tp in ["RH","LH","RF","LF","H"]:
#        detection[i0][tp] = {}
        time = list_pos["Time (s)"]
        n = time.shape[0]
        dt = np.mean(np.diff(time))

        p = np.zeros((n,2))
        p[:,0] = list_pos[tp+"x (pix)"]
        p[:,1] = list_pos[tp+"y (pix)"]
        pos[tp] = p.copy()

    i = 0
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(video_out), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_width}x{frame_height}",
        "-r", str(fps),
        "-i", "-",  # input from stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        video_out
    ]

    # Start ffmpeg subprocess
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

#    if ".avi" in video_out:
#        output = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('M','J','P','G'),fps, (frame_width, frame_height))
#    elif ".mp4" in video_out:
#        output = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),fps, (frame_width, frame_height))
#    else:
#        output = None
#    if output is not None:
    if 1:
        while(True):
            ret, frame = cap.read()
            if not ret or i>=pos['LF'].shape[0]:
                break

            cv2.circle(frame,(int(pos['LF'][i,0]),int(pos['LF'][i,1])), 10, (0,255,0), -1)
            cv2.circle(frame,(int(pos['RF'][i,0]),int(pos['RF'][i,1])), 10, (0,255,0), -1)
            cv2.circle(frame,(int(pos['RH'][i,0]),int(pos['RH'][i,1])), 10, (0,0,255), -1)
            cv2.circle(frame,(int(pos['LH'][i,0]),int(pos['LH'][i,1])), 10, (0,0,255), -1)
            cv2.circle(frame,(int(pos['H'][i,0] ),int(pos['H'][i,1] )), 10, (0,255,255), -1)
            for j in range(i):
                cv2.circle(frame,(int(pos['H'][j,0] ),int(pos['H'][j,1] )), 2, (0,255,255), -1)
            proc.stdin.write(frame.tobytes())

            i += 1

        cap.release()
#        output.release()
    proc.stdin.close()
    proc.wait()
    return
