import numpy as np
import colorsys
from PersonTracking.util import boundingBox2FaceImage
from FaceDetection.FaceDetection import FaceDetection

class TrackedPerson:
    """ 
    Person object 
    
    parameters
    face_detection_sequence: list of FaceDetections
    frame_images (list of numpy images): video frames
    small_face_size (int,int): Size of small face crops
    large_face_size (int,int): Size of large face crops
    face_padding (float): added padding percentagefor large faces
    discard_percentage (float): throw away (minimize confidence) this percentage of faces with the lowest detection confidence (faces are discarded separately for first frames and spaced frames).
    """

    def __init__(self, 
                 face_detection_sequence,
                 frame_images,
                 small_face_size=(160,160),
                 large_face_size=(299,299),
                 face_padding=0.15,
                 n_first_frames=10,
                 discard_percentage=0.0):
        assert len(frame_images)==len(face_detection_sequence), "Number of frames is different from detections"
        self.face_detection_sequence = face_detection_sequence
        self.small_face_size = small_face_size
        self.large_face_size = large_face_size
        self.n_first_frames = n_first_frames
        self.discard_percentage = discard_percentage
        self.faceEmbeddings, self.small_faces_array = None,None
        self.large_faces_array, self.raw_faces_list = None,None
        self._cropFaces(frame_images)
        self.throwAwayLessConfidentFaces()

    def throwAwayLessConfidentFaces(self):
        """ Called once on init to discard bad face detections """
        # get confidence scores or weights
        scores = self.getWeights()
        first_scores = scores[:self.n_first_frames]
        spaced_scores = scores[self.n_first_frames:]

        # how many to discard?
        n_throw_away_first = int(self.discard_percentage * len(first_scores))
        n_throw_away_spaced = int(self.discard_percentage * len(spaced_scores))

        # Get their indices
        first_discard_indices = np.array(first_scores).argsort()[:n_throw_away_first]
        spaced_discard_indices = np.array(spaced_scores).argsort()[:n_throw_away_spaced]
        # combine indices to a list
        discard_indices = list(first_discard_indices) + list(spaced_discard_indices + self.n_first_frames)

        # minimize their weight instead of discarding completely
        for ind in discard_indices:
            self.face_detection_sequence[ind].score *= 0.1


    def _cropFaces(self, frame_images):
        """ Called once on init to crop face images from frames """
        # init face arrays and list, different models take different face sizes
        self.n_total_frames = len(self.face_detection_sequence)
        self.small_faces_array = np.zeros((self.n_total_frames, self.small_face_size[0], self.small_face_size[1],3), dtype=np.uint8)
        self.large_faces_array = np.zeros((self.n_total_frames, self.large_face_size[0], self.large_face_size[1],3), dtype=np.uint8)
        self.raw_faces_list = []
        
        # extract faces
        for i,bb in enumerate(self.getBoundingBoxes()):
            self.small_faces_array[i] = boundingBox2FaceImage(np_img=frame_images[i], bb=bb, face_size=self.small_face_size, aspect_resize=False)
            self.large_faces_array[i] = boundingBox2FaceImage(np_img=frame_images[i], bb=bb, face_size=self.large_face_size, aspect_resize=True)
            self.raw_faces_list.append(boundingBox2FaceImage(np_img=frame_images[i], bb=bb, rawFace=True))

    def getBoundingBoxes(self):
        return np.array([face_det.bb() for face_det in self.face_detection_sequence])

    def getLandmarks(self):
        return np.array([face_det.landmarks() for face_det in self.face_detection_sequence])

    def getLandmarkSamples(self):
        samples_list = []
        for landmarks, raw in zip(self.getLandmarks(),self.raw_faces_list):
            samples = []
            for point in landmarks:
                x = np.clip(point[1], 0, raw.shape[0]-1)
                y = np.clip(point[0], 0, raw.shape[1]-1)
                rgb = raw[int(x),int(y),:] / 255.0
                samples.append(colorsys.rgb_to_hsv(*rgb))
            samples_list.append(samples)
        return samples_list

    def getWeights(self):
        """ Returns confidence weights for each face detection """
        return [facedet.score for facedet in self.face_detection_sequence]

    def getData(self):
        """ Returns a list of face embeddings, small faces, large faces, and raw faces"""
        return [self.faceEmbeddings,
                self.small_faces_array,
                self.large_faces_array,
                self.raw_faces_list,
                self.getLandmarks(),
                self.getLandmarkSamples(),
                self.getWeights()
                ]

    
