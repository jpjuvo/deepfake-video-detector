import numpy as np
import cv2
import PersonTracking
from PersonTracking.MultiPersonFaceTracker import MultiPersonFaceTracker

def extractPersons(face_dets_list, 
                   average_person_count, 
                   np_imgs,
                   small_face_size=(160,160),
                   large_face_size=(299,299),
                   face_padding=0.15,
                   n_first_images=10, 
                   face_discard_percentage=0.3):
    """ Extract tracked persons from face detections. Returns a list of TrackedPerson """
    
    def _extract_person_dets(nestedlist, subindex):
        return [lst[subindex]for lst in nestedlist]

    tracker = MultiPersonFaceTracker(num_persons=average_person_count)
    face_dets_sequence = [tracker.trackFaces(face_dets_list[i]) for i in range(len(face_dets_list))]
    
    num_persons = tracker.getNumPersons()
    trackedPersons = [PersonTracking.TrackedPerson.TrackedPerson(_extract_person_dets(face_dets_sequence,i), 
                                                                 np_imgs,
                                                                 small_face_size=small_face_size,
                                                                 large_face_size=large_face_size,
                                                                 face_padding=face_padding,
                                                                 n_first_frames=n_first_images,
                                                                 discard_percentage=face_discard_percentage) for i in range(num_persons)]
    
    del tracker

    return trackedPersons

def boundingBox2FaceImage(np_img, bb, padding_percentage=0, face_size=(160,160), aspect_resize=False, rawFace=False):
        """
        Parameters:
        np_img : numpy image
        bb (numpy array): bounding boxe
        face_size (size tuple of ints): returned face image size
        aspect_resize (boolean): Resize keeping the aspect ratio. Gets more padding to the smaller dimension. Default=False.
        
        Returns:
        face_image (numpy array)
        """
        if rawFace:
            return np_img[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2]),:]
        w = bb[2]-bb[0]
        h = bb[3]-bb[1]
        pad_0 = int(round(w*padding_percentage))
        pad_1 = int(round(h*padding_percentage))
        if aspect_resize:
            if (w > h): # pad more height
                pad_1 += (w-h)//2
            else:
                pad_0 += (h-w)//2
        return cv2.resize(np_img[max(0,int(bb[1] - pad_1)):min(np_img.shape[0], int(bb[3] + pad_1)),
                                 max(0,int(bb[0] - pad_0)):min(np_img.shape[1],int(bb[2] + pad_0)),
                                 :],(face_size))
    