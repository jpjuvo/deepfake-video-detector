import numpy as np
import torch
# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import InceptionResnetV1
from Util.Timer import Timer

class FaceEmbeddings:
    """
    Returns face embeddings or features from sequence of faces.
    Takes in 160x160 MTCNN face crop images.
    """
    def __init__(self, 
                 device,
                 n_first_frames, 
                 verbose=0):
        self.device = device
        self.resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=self.device).eval()
        self.n_first_frames = n_first_frames
        self.verbose = verbose
        print("Loaded pytorch facenet face embeddings model.")

    def getEmbeddings(self, faces_array):
        faces_array = np.swapaxes(np.swapaxes(faces_array,1,2),1,3) # n_frames * num_people, c, w, h
        faces_array = ((faces_array.astype(np.float32) - 127.5)/ 128) #normalize to -1,1 range
        tensors = self.resnet(torch.from_numpy(faces_array).to(self.device))
        np_tensors = tensors.detach().cpu().numpy()
        del tensors
        return np_tensors

    def getFeatNames(self):
        FF_names = ["small_face_embedding_max_centroid_diff",
                    "small_face_embedding_max_consecutive_diff",
                    "small_face_embedding_max_spaced_consecutive_diff",
                    "small_face_embedding_std_centroid_diff"]
        return FF_names

    def getFaceEmbeddingFeatures(self, embeddings):
        """
        Parameters
        embeddings:                     numpy array of facenet embeddings
        n_first_frames:                 number of faces that were taken from consecutive frames

        Returns a list of
        max_diff:                       max embedding distance from the centroid face embedding
        max_grad_from_consecutives:     max difference between two consecutive first fames face embeddings
        max_grad_from_spaced_samples:   max difference between two consecutive spaced frames face embeddings
        std:                            standard deviation of embedding centroid distances 
        """
        timer = Timer()

        n_frames_total = embeddings.shape[0]
        max_grad_from_consecutives = 0
        max_grad_from_spaced_samples = 0

        for j in range(1,self.n_first_frames):
            max_grad_from_consecutives = max(max_grad_from_consecutives,
                                                np.linalg.norm(embeddings[j-1] -embeddings[j]))
        for j in range(self.n_first_frames + 1, n_frames_total):
            max_grad_from_spaced_samples = max(max_grad_from_spaced_samples,
                                                np.linalg.norm(embeddings[j-1] -embeddings[j]))
        center = np.mean(embeddings, axis=0)
        dist = np.linalg.norm(embeddings-center, axis=1)

        max_diff = np.max(dist)
        std = np.std(dist)

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)
        return [max_diff, max_grad_from_consecutives, max_grad_from_spaced_samples, std]