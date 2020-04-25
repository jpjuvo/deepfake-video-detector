import numpy as np
import math
from FaceDetection.FaceDetector import FaceDetection

class MultiPersonFaceTracker:
    """
    Detecs the number of persons present from the first frame. Currently, detecs one or two persons.
    Keeps track of detections and returns the same number of faces from the following frames.
    """
    def __init__(self, 
                 num_persons=2,
                 multiperson_confidence_th=0.85,
                 face_confidence_decay=0.3):
        """
        Location based face tracker for scenes small number of persons.

        num_persons (int) : limit the maximum number of persons. Initial frame will lock this value to num_persons or lower
        multiperson_confidence_th (float) : in multiperson case, leave only detections over multiperson_confidence_th
        face_confidence_decay (float) : if face from previous frame is not located, the face location is copied and confidence decayed by this value

        """
        assert num_persons >= 1, "There has to be at least one person to track"
        assert num_persons <= 2, "This tracker can't track more than two persons"
        self.num_persons = num_persons
        self.prev_frame_face_dets = []
        self.firstFrame = True
        self.maxFaceDimension = 0
        self.multiperson_confidence_th = multiperson_confidence_th
        self.faceConfidenceDecay = face_confidence_decay
    
    def _getCenter(self, bb):
        return ((bb[0]+bb[2])//2, (bb[1]+bb[3])//2)

    def _getDistance(self, face_det_1,face_det_2):
        c1 = self._getCenter(face_det_1.bb())
        c2 = self._getCenter(face_det_2.bb())
        return math.sqrt(pow(c1[0]-c2[0],2) + pow(c1[1]-c2[1],2))

    def _faceTracking(self, frame_face_dets, prev_frame_face_dets):
        # match bbs to prev_bbs
        
        # len <= prev len
        if(len(prev_frame_face_dets)<=1):
            return frame_face_dets
        
        new_frame_face_dets = []
        face_det_indices = [0,1] if (self._getDistance(frame_face_dets[0], prev_frame_face_dets[0]) < 
                                     self._getDistance(frame_face_dets[0], prev_frame_face_dets[1])) else [1,0]
        face_det_indices = [face_det_indices[i] for i in range(len(frame_face_dets))]
        for i in range(len(prev_frame_face_dets)):
            if i in face_det_indices: # positional match for previous frame's face exists 
                new_frame_face_dets.append(frame_face_dets[face_det_indices.index(i)])
            else: # no positional match - use the previous frame's location and decay the confidence 
                new_frame_face_dets.append(prev_frame_face_dets[i])
                new_frame_face_dets[-1].score *= self.faceConfidenceDecay
        return new_frame_face_dets

    def _firstFrameValidation(self, frame_face_dets):
        """
        To avoid false positive second persons, see that all faces have high confidence.
        """
        if len(frame_face_dets) > 1:
            new_frame_face_dets = []
            highest_score = 0
            most_confident_face_det = None

            # in multiperson case, leave only detections over multiperson_confidence_th
            for face_det in frame_face_dets:
                if face_det.score > highest_score:
                    highest_score = face_det.score
                    most_confident_face_det = face_det
                if face_det.score > self.multiperson_confidence_th:
                    new_frame_face_dets.append(face_det)

            # if the threshold ruled out all candidates, leave the most confident
            if len(new_frame_face_dets)==0:
                new_frame_face_dets.append(most_confident_face_det)
            return new_frame_face_dets
        return frame_face_dets

    def trackFaces(self, frame_face_dets):
        if len(frame_face_dets) > self.num_persons:
            frame_face_dets = frame_face_dets[:self.num_persons]
        
        # keep the number of persons in the first frame
        if self.firstFrame:
            self.firstFrame = False
            frame_face_dets = self._firstFrameValidation(frame_face_dets)
            self.num_persons = min(self.num_persons,max(len(frame_face_dets),1))
        else:
            # keep the same face order and always find the same num_persons
            frame_face_dets = self._faceTracking(frame_face_dets, self.prev_frame_face_dets)

        # record the maximum dimension
        for face_det in frame_face_dets:
            self.maxFaceDimension = max(self.maxFaceDimension, face_det.height())
            self.maxFaceDimension = max(self.maxFaceDimension, face_det.width())
        
        self.prev_frame_face_dets = frame_face_dets
        return frame_face_dets

    def getNumPersons(self):
        return self.num_persons

    def getMaxFaceDimension(self):
        return self.maxFaceDimension
