import os
import glob
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
import moviepy.editor as mp
import scipy.io.wavfile as wav
from tqdm.notebook import tqdm
from pydub import AudioSegment

with open("../user_config.json") as config:
    path_dict = json.load(config)['data_paths']
    AUDIO_DIR = path_dict['audio_path']

def video2wav(filename, AUDIO_DIR=AUDIO_DIR):
    filename = str(filename)
    audio_sample = os.path.join(AUDIO_DIR, filename.split("/")[-1].replace(".mp4",".wav"))
    if(os.path.isfile(audio_sample)):
        return
    try:
        clip = mp.VideoFileClip(filename)
        audio = clip.audio
        audio.write_audiofile(audio_sample,verbose=False,logger=None)
    except:
        print("Except in " + audio_sample)
