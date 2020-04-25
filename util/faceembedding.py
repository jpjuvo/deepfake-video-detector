import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import math

# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceEmbedding:

    def __init__(self, every_n_frame=30, downsampling=4):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if(self.device == 'cpu'):
            raise Exception('Cuda is required')

        # Load face detector
        self.mtcnn = MTCNN(keep_all=True, select_largest=False, device=self.device).eval()

        # Load facial recognition model
        self.resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=self.device).eval()

        self.n_frames = every_n_frame
        self.downsampling = downsampling

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

    def _getFaceCrops(self, img, bbs, face_size=(160,160), brightness=0,contrast=1.3):
        """
        Parameters:
        img (PIL image): image
        bbs (numpy array): bounding boxes
        face_size (size tuple of ints): returned face image size
        
        Returns:
        faces (numpy arrays)
        """
        imgs = []
        np_img = np.clip((np.array(img)+brightness)*contrast,0,255).astype(np.uint8)
        for bb in bbs:
            imgs.append(cv2.resize(np_img[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2]),:],(face_size)))
        return imgs

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

    def _embeddings(self, faces_array):
        # input is 10 frames * num_people, h, w, c
        faces_array = np.swapaxes(np.swapaxes(faces_array,1,2),1,3) # 10 frames * num_people, c, w, h
        faces_array = (faces_array.astype(np.float32) / 255) - 0.5 #normalize
        return self.resnet(torch.from_numpy(faces_array).to(self.device))

    def _getFaceBBs(self, imgs):
        full_size = imgs[0].size
        ds_imgs = [img.resize((full_size[0]// self.downsampling, full_size[1]// self.downsampling)) for img in imgs]
        face_dets_list, scores_list = self.mtcnn.detect(ds_imgs)
        face_dets_list = [face_det * self.downsampling for face_det in face_dets_list]
        return face_dets_list, scores_list

    def getFaceEmbeddings(self, videoPath, return_images=True):
        # Create video reader and find length
        v_cap = cv2.VideoCapture(str(videoPath))
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        imgs = []
        for i,j in enumerate(range(0,v_len)):
            success, vframe = v_cap.read()
            if i%self.n_frames==0:
                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                imgs.append(Image.fromarray(vframe))
        v_cap.release()
        n_imgs = v_len//self.n_frames

        num_faces = 2 # 1 or 2
        prev_bbs = []

        face_dets_list, scores_list = self._getFaceBBs(imgs)
        faces_array = np.zeros((2 * n_imgs, 160,160,3), dtype=np.uint8)

        for i in range(len(imgs)):
            face_dets, scores = face_dets_list[i], scores_list[i]
            bbs = self._removeOverlappingDetections(face_dets, scores, n_max=num_faces)
            
            # set and keep the num_faces from the first frame
            if(i==0):
                num_faces = max(len(bbs),1)
            else:
                # keep the same face order and always find the same num_faces
                bbs = self._faceTracking(bbs, prev_bbs)
            
            prev_bbs = bbs
            
            # crop to 160x160
            faces = self._getFaceCrops(imgs[i], bbs)
            
            # add for list to get embeddings later
            for j, face in enumerate(faces):
                faces_array[i + j * (n_imgs)] = np.array(face)

        embedding = self._embeddings(faces_array)

        return_images = []
        return_embeddings = []

        if num_faces >= 1:
            return_images.append(faces_array[:n_imgs])
            return_embeddings.append(embedding[:n_imgs])
        
        if num_faces==2: # add second set of faces
            return_images.append(faces_array[n_imgs:])
            return_embeddings.append(embedding[n_imgs:])

        if return_images:
            return return_embeddings, return_images
        
        return return_embeddings

