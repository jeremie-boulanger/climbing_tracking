import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import torch
import sys
import re
from scipy import interpolate
from scipy.signal import butter, filtfilt, medfilt
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
import streamlit as st
import os
import subprocess
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

def yolo_tracking(video_file, tracking_yolo, out_image_name=None, heatmap_image_name=None, model_name='yolov8l.pt', title=None, display=True, stream=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.write("Error opening video stream or file")
        raise TypeError
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, frame0 = cap.read() # Just for the background of the plot
    cap.release()

    st.write("Video frame size:", frame_width, frame_height)
    st.write("Video duration:", length, "frames")

    model = YOLO(model_name).to(device)
    st.write(f"Device to run the model: {model.device}") 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tracker_path = os.path.join(current_dir, "custom_tracker.yaml")

    results = model.track(source=video_file, classes=[0], show=False, verbose=False, stream=stream, tracker=tracker_path)
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
        fig, ax = plt.subplots(dpi=100)
        ax.imshow(cv2.cvtColor(frame0,cv2.COLOR_BGR2RGB))
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
        myfig = plt.figure(dpi=100)
        plt.imshow(cv2.cvtColor(frame0,cv2.COLOR_BGR2RGB))
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
                plt.plot(lx, [frame_height - y for y in ly], label=str(idx))
                if max(ly) > m:
                    m = max(ly)
                    tracking_person2 = tracking_person
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

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

    # ---------- MediaPipe PoseLandmarker ----------
    MODEL_PATH = "./mediapipe_models/pose_landmarker_heavy.task"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.9
    )

    landmarker = PoseLandmarker.create_from_options(options)

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
                image_crop = image[int(b[1]):int(b[3]),int(b[0]):int(b[2]),:].astype(np.uint8)

                if image_crop.size == 0:
                    break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_crop)
                result = landmarker.detect(mp_image)
                if not result.pose_landmarks:
                    #t_idx += 1
                    continue
                rp = result.pose_landmarks
                rpw = result.pose_world_landmarks
                if rp is not None:
                    lrp = [{'x':b[0]+(b[2]-b[0])*l.x,'y':b[1]+(b[3]-b[1])*l.y, 'z':l.z, 'visib':l.visibility} for l in rp[0]]
                    lrpw= [{'x':l.x,'y':l.y, 'z':l.z, 'visib':l.visibility} for l in rpw[0]]
                    list_pose.append({'timestamp':list_detection[t_idx]['timestamp'],'pose':lrp, 'world pose':lrpw, 'size':(width, height)})
                break
        t_idx += 1

    landmarker.close()
    cap.release()

    save = True # Saving the tracking
    if save:
        os.makedirs(os.path.dirname(tracking_initial), exist_ok=True)
        with open(tracking_initial, 'wb') as handle:
            pickle.dump(list_pose, handle)
    bar_mediapipe.empty()
    return

def extract_tracking(tracking_initial, pickle_tracking_file, xlsx_tracking_file, w_c, dx, dy):
    median_win = 21
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


    list_body_indexes = {"H": [23,24], "LH": [15,21,19,17], "RH": [16,22,20,18], "LF": [27,29,31], "RF": [28,30,32]}
    data = {'Time (s)':time2/1000.}


    for l in list_body_indexes:

        # determine the position
        pos = np.array([np.array([[l['pose'][i]['x'],l['pose'][i]['y']] for l in list_pose]) for i in list_body_indexes[l]])
        pos = np.mean(pos,axis=0)
        for j in range(pos.shape[1]):
            pos[:,j] = medfilt(pos[:,j], median_win)   

        # determine the visibility
        visib = np.array([np.array([[l['pose'][i]['visib']] for l in list_pose]) for i in list_body_indexes[l]])
        visib = np.clip(np.mean(visib,axis=0),0.2,1) # clipping to avoid non stability

        # interpolate and low pass filter
        f = interpolate.interp1d(time,pos,axis=0)
        pos = f(time2)
        f = interpolate.interp1d(time,visib,axis=0)
        visib = f(time2)[:,0]
        for j in range(pos.shape[1]):
            pos[:,j] = filtfilt(b, a, pos[:,j] * visib, axis=0) / filtfilt(b, a, visib, axis=0)
        visib = filtfilt(b, a, visib, axis=0)
        
        data[l+"x (pix)"] = np.clip(pos[:,0],0,width-1)
        data[l+"y (pix)"] = np.clip(pos[:,1],0,height-1)
        
        
        data[l+' visib'] = visib[:]
        

        # compute the velocity
        v = np.linalg.norm(np.diff(pos,axis=0),axis=1)/dt
        v = np.insert(v,0,v[0])

        data[l+"v (pix/s)"] = v[:]

        pos[:,0] *= dx
        pos[:,1] *= dy
        v = np.linalg.norm(np.diff(pos,axis=0),axis=1)/dt
        v = np.insert(v,0,v[0])
        data[l+"x (m)"] = np.clip(pos[:,0],0,width-1)
        data[l+"y (m)"] = np.clip(pos[:,1],0,height-1)
        data[l+"v (m/s)"] = v[:]
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
    while(True):
        ret, frame = cap.read()
        if not ret or i>=len(list_pos["LFx (pix)"]):
            break
        cv2.circle(frame,(int(list_pos["LFx (pix)"][i]),int(list_pos["LFy (pix)"][i])), int(15*list_pos["LF visib"][i]), (255,255,0), -1)
        cv2.circle(frame,(int(list_pos["RFx (pix)"][i]),int(list_pos["RFy (pix)"][i])), int(15*list_pos["RF visib"][i]), (0,255,0), -1)
        cv2.circle(frame,(int(list_pos["LHx (pix)"][i]),int(list_pos["LHy (pix)"][i])), int(15*list_pos["LH visib"][i]), (0,0,255), -1)
        cv2.circle(frame,(int(list_pos["RHx (pix)"][i]),int(list_pos["RHy (pix)"][i])), int(15*list_pos["RH visib"][i]), (255,0,255), -1)
        cv2.circle(frame,(int(list_pos["Hx (pix)"][i]) ,int(list_pos["Hy (pix)"][i])) , int(15*list_pos["H visib"][i]) , (0,255,255), -1)
        for j in range(i):
            cv2.circle(frame,(int(list_pos["Hx (pix)"][j]),int(list_pos["Hy (pix)"][j])), 3, (0,255,255), -1)            
        proc.stdin.write(frame.tobytes())
        i += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    return
