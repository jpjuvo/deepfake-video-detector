import cv2
import os
import PIL
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
from Util.Timer import Timer

class VideoFrameSampler:
    """Sample frames from a video starting by collecting n_first_frames,
    and the sampling n_spaced_frames evenly from the rest of the video.
    getFrames Returns list of PIL images"""

    def __init__(self,
                 n_first_frames=10,
                 n_spaced_frames=10,
                 low_light_th=60,
                 max_video_frames_to_read=300, # at 30 fps, this equals 10 first seconds of the video 
                 verbose=0
                 ): 
        self.n_first_frames = n_first_frames
        self.n_spaced_frames = n_spaced_frames
        self.low_light_th = low_light_th
        self.max_video_frames_to_read = max_video_frames_to_read
        self.verbose = verbose
        print("Loaded video frame sampler.")

    def getFrames(self, videoPath, frame_offset, frame_end_offset, first_frame_step=1,increase_lightness=False, override_brightness_fc=None):
        timer = Timer()

        assert os.path.isfile(videoPath), "{0} not found".format(videoPath)

        v_cap = cv2.VideoCapture(str(videoPath))
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # limit the reading length
        if self.max_video_frames_to_read > 0:
            v_len = min(v_len,self.max_video_frames_to_read) - frame_end_offset

        # check that the offset doesn't limit sample count
        if v_len <= (frame_offset + self.n_first_frames * first_frame_step + self.n_spaced_frames + 1):
            frame_offset = v_len - self.n_first_frames * first_frame_step - self.n_spaced_frames - 2

        assert frame_offset >= 0, "There are not enough frames to sample or the reading offset is smaller than zero"

        imgs = []
        frame_brightnesses = []
        
        capture_frame_indices = list(range(frame_offset, frame_offset + self.n_first_frames * first_frame_step, first_frame_step))
        capture_frame_indices += list(np.linspace(frame_offset + self.n_first_frames * first_frame_step, 
                                                  v_len - 1, 
                                                  self.n_spaced_frames + 1, 
                                                  endpoint=True, 
                                                  dtype=np.int))[1:]

        if self.verbose > 1:
            print("Sampling frame indices: {0}".format(capture_frame_indices))

        for i in range(v_len):
            # read frame but don't encode it if it's not needed
            if not v_cap.grab():
                continue

            current = len(imgs)
            if i == capture_frame_indices[current]:
                ret, frame = v_cap.retrieve()
                if not ret or frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imgs.append(PIL.Image.fromarray(frame)) # mtcnn takes PIL images
                frame_brightnesses.append(np.mean(frame))
                if current == len(capture_frame_indices)-1:
                    break
            elif i>capture_frame_indices[current]:
                break
        
        v_cap.release()

        # brighten dark video
        mean_brightness = np.mean(np.array(frame_brightnesses))
        brightness_fc = None
        if mean_brightness < self.low_light_th or increase_lightness or override_brightness_fc is not None:
            brightness_fc = min(max(2,self.low_light_th / mean_brightness),1.2) if not increase_lightness else 1.5
            if override_brightness_fc is not None:
                brightness_fc = override_brightness_fc
            imgs = [ImageEnhance.Brightness(img).enhance(brightness_fc) for img in imgs]

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)

        return imgs, brightness_fc
