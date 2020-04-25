import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
from pathlib import Path
import warnings

with open("../user_config.json") as config:
    path_dict = json.load(config)['data_paths']
    SAVE_DIR = path_dict['spectrogram_path']

def wav2spectrogram(filename, SAVE_DIR=SAVE_DIR):
    AUDIO_FN_EXTENSION = ".wav"
    spectogram_sample = os.path.join(SAVE_DIR, str(filename).split("/")[-1].replace(".wav",".png"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            samples, sample_rate = librosa.load(str(filename))
            fig = plt.figure(figsize=[0.72,0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

            S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            plt.savefig(spectogram_sample, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')
            
        except:
            print("Except in " + spectogram_sample)