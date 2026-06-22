import cv2
import subprocess
import pickle
import os
import numpy as np
import pandas as pd

def segmentation(video_file, tracking_file, tracking_file_xlsx, folder_hold, save = True) :
    """
    Detects movement intervals from tracking data and cut the video into several trial.
    """
    threshold = find_white_pixel(folder_hold)
    cap = cv2.VideoCapture(video_file)
    cap.release()

    with open(tracking_file,'rb') as o:
        list_pos = pickle.load(o)

    df_xlsx = pd.read_excel(tracking_file_xlsx)
    band = search_band(list_pos["RFy (pix)"], threshold)
    i = 1
    lst_essaie = []

    for k in band :
        start_time = k[0]
        end_time = k[1]
        
        # Retire les vidéos inférieur à 5 secondes
        if list_pos["Time (s)"][end_time] - list_pos["Time (s)"][start_time] > 5:

            # Segmentation Vidéo
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-ss", str(max(0, list_pos["Time (s)"][start_time])),
                "-i", video_file,
                "-t", str(list_pos["Time (s)"][end_time] - max(0, max(0, list_pos["Time (s)"][start_time]))),
                "-map", "0:v",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                video_file[:-4] + f"_E{i}.mp4"
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # Segmentation .pickle
            if save :
                pickle_lst = {key: (value[start_time:end_time+1] if isinstance(value, (list, np.ndarray)) else value) for key, value in list_pos.items()}
                with open(tracking_file[:-7] + f"_E{i}.pickle", 'wb') as f:
                    pickle.dump(pickle_lst, f)
                    
            # Segmentation .xlsx
            if save :
                xlsx = df_xlsx.iloc[start_time:end_time+1]
                xlsx.to_excel(tracking_file_xlsx[:-5] + f"_E{i}.xlsx", index=False)


            lst_essaie.append((video_file[:-4] + f"_E{i}.mp4",tracking_file[:-7] + f"_E{i}.pickle"))
            i += 1
    
    return lst_essaie
   
def find_white_pixel(folder_hold) :
    """
    Combines all images in a folder and identifies white pixels. Returns the highest vertical position where a white pixel is detected.
    """
    current_image = None
    for f in os.listdir(folder_hold):
        path = os.path.join(folder_hold, f)
        if not folder_hold :
            raise FileNotFoundError(f"Error opening image or file")

        if current_image is None :
              current_image = cv2.imread(path)
        else :
            new_image = cv2.imread(path)
            current_image = cv2.addWeighted(current_image, 1, new_image, 1, 0)

    gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    white_pixel = np.where(gray > 200)
    max_y = np.max(white_pixel[0])
    print(max_y)
    return max_y

def search_band(data, threshold):
    """
    Finds intervals where the input data remains below a threshold. Returns the start and end indices of these intervals.
    """
    masque = data < threshold
    masque = np.concatenate(([False], masque, [False])) # pour toujours avoir un début et une fin pour chaque intervalle
    diff = np.diff(masque.astype(int))
    
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    
    return list(zip(starts, ends))

def segmentation_intervals(video_file, intervals) :
    """
    Cut a video into multiple trial based on provided intervals.
    """
    cap = cv2.VideoCapture(video_file)
    cap.release()

    for i, k in enumerate(intervals) :
        start_time = k[0]
        end_time = k[1]

        # Segmentation Video
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", video_file,
            "-t", str(end_time - start_time),
            "-map", "0:v",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            f"{video_file[:-4]}_E{i+1}.mp4"
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
