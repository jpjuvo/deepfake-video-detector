import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import math
import torch
import torch.nn as nn
from torchvision import transforms

# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1

class FFFimagesampler:

    def __init__(self, faceforensics_model_paths, n_frames=10, downsampling=2, face_embedding_size=(160,160), faceforensics_size=(299,299), face_padding=0.15, max_retries=5, min_frames=None):
        """
        Process first n consecutive frames of a video and output high-level features for a second level DeepFake classifier.
        This class detects and tracks faces of 1-2 persons and processes each persons faces for DeepFake features.
        Features include max deviation from a centroid face embedding during the sampled frames and max consecutive frame embedding distances.
        FaceForensics features include mean and max DeepFake classifier outputs for each persons faces.
        In two-person cases, maximum fetaure values are returned.

        Note, FaceNet expects to find pretrained models from /tmp/.cache/torch/checkpoints/ and downloads the weights if missing.
        To get the weights without internet, copy the weights manually by running in jupyter cell:
        !mkdir -p /tmp/.cache/torch/checkpoints/
        !cp [weight folder]/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt
        !cp [weight folder]/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt

        Parameters:
        faceforensics_model_paths (string): list of paths to pretrained faceforensics models. Download pretrained models from http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip
        n_frames (int): Number of consecutive frames to sample (affects the output feature qualities and processing times). Default=10.
        downsampling (int): Video dowsampling factor for the Face detection model for faster processing (2 works well with HD videos but higher factors may miss more faces). Doesn't affect anything else. Default=2.
        face_embedding_size (size int tuple): FaceNet face recognition model. Pretrained model is trained with (160,160) size, default=(160,160).
        faceforensics_size (size int tuple): FaceForensics++ model's input size. Default=(299,299).
        face_padding (float): x and y padding percentage of width or height that is added on both sides for the FaceForensics++ face inputs. Faceforensics++ paper enlarged face crops 1.3 times. Default=0.15.
        max_retries (int): Number of times to retry if less than min_faces faces are detected from processed frames. Each retry samples from the following frames. Default=5.
        min_frames (int): If not None, min frames before retry. If None, min_frames is n_frames.
        """
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if(self.device == 'cpu'):
            raise Exception('Cuda is required')

        # Face detector model - finds face bounding boxes
        self.mtcnn = MTCNN(keep_all=True, select_largest=False, device=self.device).eval()

        # FaceForensics face image preprocessing
        self.xception_default_data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(faceforensics_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3)
            ]),
        }

        self.n_frames = n_frames
        self.downsampling = downsampling
        self.post_function = nn.Softmax(dim=1)
        self.faceforensics_size = faceforensics_size
        self.face_embedding_size = face_embedding_size
        self.face_padding = face_padding
        self.max_retries = max_retries
        self.min_frames = min_frames if min_frames is not None else n_frames

    #############################################################
    ###################  Face Detection #########################

    def _getFaceBBs(self, imgs):
        full_size = imgs[0].size
        ds_imgs = [img.resize((full_size[0]// self.downsampling, full_size[1]// self.downsampling)) for img in imgs]
        face_dets_list, scores_list = self.mtcnn.detect(ds_imgs)
        try:
            face_dets_list = [face_det * self.downsampling for face_det in face_dets_list]
        except:
            return [],[]
        return face_dets_list, scores_list

    #############################################################
    ############### Face tracking and postprocessing ############

    def _removeOverlappingDetections(self, bbs, scores, n_max=2):
        """Filter out the less confident overlapping bounding boxes 
        and discard the least confident extra detections (>n_max)"""
        # don't do anything with one or less detection
        if(len(bbs)<=1):
            return bbs
        
        remove_list = []
        #remove overlapping
        for i in range(len(bbs)):
            for j in range(i + 1,len(bbs)):
                # detections are ordered the largest first
                # get leftmost, rightmost
                LM,RM = (i,j) if (bbs[i][0] < bbs[j][0]) else (j,i)
                if (bbs[RM][0] < bbs[LM][2]): # overlapping horizontally
                    # get topmost bottommost
                    TM, BM = (i,j) if (bbs[i][1] < bbs[j][1]) else (j,i)
                    if (bbs[BM][1] < bbs[TM][3]): # overlapping vertically
                        remove_list.append(j) # use the original order to judge importance
                        
        # return filtered and only n_max bbs at maximum 
        keep_bbs = [bbs[i] for i in range(len(bbs)) if i not in remove_list]
        return keep_bbs[:min(len(keep_bbs),n_max)]

    def _getCenter(self, bb):
        return ((bb[0]+bb[2])//2, (bb[1]+bb[3])//2)

    def _getDistance(self, bb1,bb2):
        c1 = self._getCenter(bb1)
        c2 = self._getCenter(bb2)
        return math.sqrt(pow(c1[0]-c2[0],2) + pow(c1[1]-c2[1],2))

    def _faceTracking(self, bbs, prev_bbs):
        # match bbs to prev_bbs
        # len(bbs) <= prev_bbs
        if(len(prev_bbs)<=1):
            return bbs
        
        new_bbs = []
        bbs_indices = [0,1] if self._getDistance(bbs[0], prev_bbs[0]) < self._getDistance(bbs[0], prev_bbs[1]) else [1,0]
        bbs_indices = [bbs_indices[i] for i in range(len(bbs))]
        for i in range(len(prev_bbs)):
            if i in bbs_indices:
                new_bbs.append(bbs[bbs_indices.index(i)])
            else:
                new_bbs.append(prev_bbs[i])
        return new_bbs

    def _getFaceCrops(self, img, bbs, padding_percentage=0, face_size=(160,160), aspect_resize=False):
        """
        Parameters:
        img (PIL image): image
        bbs (numpy array): bounding boxes
        face_size (size tuple of ints): returned face image size
        aspect_resize (boolean): Resize keeping the aspect ratio. Gets more padding to the smaller dimension. Default=False.
        
        Returns:
        faces (numpy arrays)
        """
        imgs = []
        np_img = np.array(img, dtype=np.uint8)
        for bb in bbs:
            w = bb[2]-bb[0]
            h = bb[3]-bb[1]
            pad_0 = int(round(w*padding_percentage))
            pad_1 = int(round(h*padding_percentage))
            if aspect_resize:
                if (w > h): # pad more height
                    pad_1 += (w-h)//2
                else:
                    pad_0 += (h-w)//2
            imgs.append(cv2.resize(np_img[
                max(0,int(bb[1] - pad_1)):min(np_img.shape[0], int(bb[3] + pad_1)),
                max(0,int(bb[0] - pad_0)):min(np_img.shape[1],int(bb[2] + pad_0)),
                :],(face_size)))
        return imgs

    #############################################################
    ####################### MAIN ################################

    def Predict(self, videoPath, n_frames, return_embeddings=False, frame_offset=0, retries=0):
        """
        Return second level features for DeepFake classification.

        Parameters:
        videoPath (string): Path to .mp4 video.
        return_embeddings (boolean): Return face embedding features. Default=True
        frame_offset (int): Start sampling from this frame. Default=0.
        retries (int): Number of retries done already. This is incremented when calling recursively on face detection errors until max_retries is reached. Default=0.

        Throws: Exception if max_retries is reached.

        Returns:
        faceforensics_features (float array of shape (2 * number of FF models)): FaceForensics++ predictions for faces 0=real 1=fake. Mean and Max of all frames and max of all persons.
        embedding_feature (float array of shape (2)): Face embedding max centroid deviation and max consecutive frame difference. Max values of all persons.
        """
        # Create video reader and find length
        v_cap = cv2.VideoCapture(str(videoPath))
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        imgs = []
        for i,j in enumerate(range(frame_offset,v_len)):
            success, vframe = v_cap.read()
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(vframe))
            if i>=n_frames-1:
                break
        v_cap.release()
        n_imgs = n_frames

        num_persons = 2
        prev_bbs = []

        face_dets_list, scores_list = self._getFaceBBs(imgs)

        # retry if less than n_frames faces
        if len(face_dets_list) < n_frames:
            if retries < self.max_retries:
                return self.Predict(videoPath, n_frames, return_embeddings=return_embeddings, frame_offset=frame_offset+n_imgs, retries=retries+1)
            else:
                raise Exception("Maximum retries with " + videoPath)

        padded_faces_array = np.zeros((2 * n_imgs, self.faceforensics_size[0], self.faceforensics_size[1],3), dtype=np.uint8)

        for i in range(len(imgs)):
            face_dets, scores = face_dets_list[i], scores_list[i]
            bbs = self._removeOverlappingDetections(face_dets, scores, n_max=num_persons)
            
            # set and keep the num_persons from the first frame
            if(i==0):
                num_persons = max(len(bbs),1)
            else:
                # keep the same face order and always find the same num_persons
                bbs = self._faceTracking(bbs, prev_bbs)
            
            prev_bbs = bbs
            
            padded_faces = self._getFaceCrops(imgs[i], bbs, padding_percentage=self.face_padding, face_size=self.faceforensics_size, aspect_resize=True)
            
            # add for list to get embeddings later
            for j, face in enumerate(padded_faces):
                padded_faces_array[i + j *(n_imgs)] = np.array(padded_faces[j])

        return padded_faces_array[:num_persons*n_imgs]