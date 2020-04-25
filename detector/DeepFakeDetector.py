import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from pathlib import Path
import random
import torch

from FaceDetection.FaceDetector import FaceDetector
from FaceDetection.FaceDetector import FaceDetectorError
from FaceDetection.FaceDetection import FaceDetection
from Util.VideoFrameSampler import VideoFrameSampler
from PersonTracking.MultiPersonFaceTracker import MultiPersonFaceTracker
from PersonTracking.TrackedPerson import TrackedPerson
from PersonTracking.util import extractPersons
from FeatureGenerators.FaceEmbeddings import FaceEmbeddings
from FeatureGenerators.FaceClassifier import FaceClassifier
from FeatureGenerators.PowerSpectrumClassifier import PowerSpectrumClassifier
from SecondLevelClassifier import SecondLevelClassifier
from FeatureGenerators.FaceSequenceClassifier import FaceSequenceClassifier
from Util.ImageUtil import JPEGCompression, ResizeImage

class VideoAugmentation:
    """ Augmentations intented for second level model training. """
    NONE = None
    HALF_FPS = 'fps_15'
    FOURTH_OF_SIZE = 'resize_smaller'
    COMPRESS = "jpeg_compression"

class DeepFakeDetector:
    
    def __version__(self):
        return "0.9.0"

    def __init__(self, 
                 deepfake_models_directory,
                 third_party_models_directory,
                 n_first_frames=10,
                 n_spaced_frames=10, 
                 downsampling=2, 
                 small_face_size=(160,160), 
                 large_face_size=(299,299), 
                 face_padding=0.15, 
                 max_retries=4, 
                 predict_on_error=0.5,
                 low_light_th=60,
                 face_discard_percentage=0.0,
                 use_power_spectrum_clf=False,
                 verbose=0):
        """
        Note, FaceNet expects to find pretrained models from /tmp/.cache/torch/checkpoints/ and downloads the weights if missing.
        To get the weights without internet, copy the weights manually by running in jupyter cell:
        !mkdir -p /tmp/.cache/torch/checkpoints/
        !cp [weight folder]/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt
        !cp [weight folder]/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt

        Parameters:
        deepfake_models_directory (str): model folder of trained classifiers
        third_party_models_directory (str): model folder of third party files such as blazeface weights
        n_first_frames (int): Number of consecutive frames to sample (affects the output feature qualities and processing times). Default=10.
        n_spaced_frames (int): Number of equally spaced frames to sample from the rest of the video after n_first_frames. Default=10.
        downsampling (int): Video dowsampling factor for the Face detection model for faster processing (2 works well with HD videos but higher factors may miss more faces). Doesn't affect anything else. Default=2.
        small_face_size (size int tuple): FaceNet face recognition model. Pretrained model is trained with (160,160) size, default=(160,160).
        large_face_size (size int tuple): Default=(299,299).
        face_padding (float): x and y padding percentage of width or height that is added on both sides. Default=0.15.
        max_retries (int): Number of times to retry if less than min_faces faces are detected from processed frames. Each retry samples from the following frames. Default=4.
        predict_on_error (float): This value gets predicted on Predict method's error. default=0.5.
        low_light_th (int): If the average brightness from the sampled frames goes below this value, all frames are brightened, range [0,255], default=60.
        face_discard_percentage (float): Percentage (0-1) of faces to drop. Dropping order comes from face detector confidence so that the least confident are dropped. Default=0.0. 
        use_power_spectrum_clf (bool): If powerspectrum classifier is used (https://arxiv.org/abs/1911.00686). By default, this is False to save inference time. 
        verbose (int): 0 = silent, 1 = print errors and warnings, 2 = print processing times of components and all errors and warnings. default=0. 
        """
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if(self.device == 'cpu'):
            raise Exception('Cuda is required')
        print(self.device)

        # locate model paths
        (small_face_model_dirs, 
         large_face_model_dirs, 
         second_level_xgb_paths, 
         second_level_logreg_paths,
         second_level_lgb_paths,
         power_spectrum_path,
         recurrent_model_dirs) = self._getModelPaths(deepfake_models_directory)

        blazeface_path, blazeface_anchors = self._getThirdPartyModelPaths(third_party_models_directory)

        self.frameSampler = VideoFrameSampler(n_first_frames=n_first_frames, 
                                              n_spaced_frames=n_spaced_frames,
                                              low_light_th=low_light_th,
                                              verbose=verbose)
        
        self.faceDetector = FaceDetector(self.device,
                                         blazeface_device='cpu',
                                         blazeface_path=blazeface_path,
                                         blazeface_anchors=blazeface_anchors,
                                         mtcnn_downsampling=downsampling,
                                         verbose=verbose)

        self.faceEmbeddings = FaceEmbeddings(self.device,
                                             n_first_frames=n_first_frames,
                                             verbose=verbose)

        self.faceClassifier = FaceClassifier(small_face_model_dirs,
                                             large_face_model_dirs,
                                             n_first_frames, 
                                             n_spaced_frames,
                                             verbose=verbose)

        self.faceSequenceClassifier = FaceSequenceClassifier(recurrent_model_dirs,
                                                             n_first_frames,
                                                             n_spaced_frames,
                                                             verbose=verbose)
        if use_power_spectrum_clf:
            self.powerSpectrumClassifier = PowerSpectrumClassifier(power_spectrum_path,
                                                                   verbose=verbose)

        self.secondLevelClassifier = SecondLevelClassifier(second_level_xgb_paths, 
                                                           second_level_logreg_paths,
                                                           second_level_lgb_paths,
                                                           verbose=verbose)

        self.n_first_frames = n_first_frames
        self.n_spaced_frames = n_spaced_frames
        self.large_face_size = large_face_size
        self.small_face_size = small_face_size
        self.face_padding = face_padding
        self.max_retries = max_retries
        self.predict_on_error = predict_on_error
        self.face_discard_percentage = face_discard_percentage
        self.use_power_spectrum_clf = use_power_spectrum_clf
        self.verbose = verbose

        self._printInfo()

    def _printInfo(self):
        print("#"*50)
        print("DeepFakeDetector v." + self.__version__())
        print("Sample {0} first frames and {1} spaced frames.".format(self.n_first_frames, self.n_spaced_frames))
        print("Number of max retries is {0}".format(self.max_retries))
        print("On error cases, predict {0}".format(self.predict_on_error))
        print("#"*50)

    def _getSubDirectories(self, dir_path):
        return [os.path.join(dir_path, o) for o in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,o))]

    def _getSubFiles(self, dir_path, suffix=None):
        return [os.path.join(dir_path, o) for o in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,o)) and (suffix in o if suffix is not None else True)]

    def _getModelPaths(self, dir_path):
        small_face_model_dirs = []
        large_face_model_dirs = []
        second_level_xgb_paths = []
        second_level_logreg_paths = []
        second_level_lgb_paths = []
        recurrent_model_dirs = []
        power_spectrum_path = None
        if dir_path is not None:
            sub_dirs = self._getSubDirectories(dir_path)
            for sub_dir in sub_dirs:
                if "small_face" in sub_dir:
                    small_face_model_dirs = self._getSubDirectories(sub_dir)
                elif "large_face" in sub_dir:
                    large_face_model_dirs = self._getSubDirectories(sub_dir)
                elif "second_level" in sub_dir:
                    date_dirs = self._getSubDirectories(sub_dir)
                    if len(date_dirs) == 0:
                        continue
                    date_dir = date_dirs[0]
                    second_level_xgb_paths = self._getSubFiles(date_dir, '.bin')
                    second_level_logreg_paths = self._getSubFiles(date_dir, '.sav')
                    second_level_lgb_paths = self._getSubFiles(date_dir, '.txt')
                elif "power_spectrum" in sub_dir:
                    date_dirs = self._getSubDirectories(sub_dir)
                    if len(date_dirs) == 0:
                        continue
                    date_dir = date_dirs[0]
                    if len(self._getSubFiles(date_dir, '.bin'))==0:
                        continue
                    power_spectrum_path = self._getSubFiles(date_dir, '.bin')[0]
                elif "recurrent_cnn" in sub_dir:
                    recurrent_model_dirs = self._getSubDirectories(sub_dir)

        return (small_face_model_dirs, 
                large_face_model_dirs, 
                second_level_xgb_paths, 
                second_level_logreg_paths,
                second_level_lgb_paths, 
                power_spectrum_path, 
                recurrent_model_dirs)

    def _getThirdPartyModelPaths(self, dir_path):
        blazeface_path = None
        blazeface_anchors = None
        if dir_path is not None:
            sub_dirs = self._getSubDirectories(dir_path)
            for sub_dir in sub_dirs:
                if "blazeface" in sub_dir:
                    blazeface_files = self._getSubFiles(sub_dir)
                    for bf_file in blazeface_files:
                        if "blazeface.pth" in bf_file:
                            blazeface_path = bf_file
                        elif "anchors.npy" in bf_file:
                            blazeface_anchors = bf_file
        return (blazeface_path, blazeface_anchors)

    def _getRandomAugmentation(self):
        """
        According to this article: https://arxiv.org/abs/1910.08854
        DFDC testing is  done with augmentations.
        This function returns
        p=2/9, HALF FPS
        p=2/9, FOURTH OF SIZE
        p=2/9, COMPRESS
        p=3/9, No augmentation 
        """
        selector = random.randint(1,9)
        if self.verbose > 0:
            print("Augmentation selector: {0}".format(selector))
        if selector < 3:
            return VideoAugmentation.HALF_FPS
        elif selector < 5:
            return VideoAugmentation.FOURTH_OF_SIZE
        elif selector < 7:
            return VideoAugmentation.COMPRESS
        else:
            return VideoAugmentation.NONE

    def _augmentPILImages(self, imgs, augmentation):
            if augmentation is None or augmentation == VideoAugmentation.HALF_FPS:
                return imgs
            if augmentation == VideoAugmentation.COMPRESS:
                return [JPEGCompression(img) for img in imgs]
            if augmentation == VideoAugmentation.FOURTH_OF_SIZE:
                return [ResizeImage(img) for img in imgs]
            raise NameError("unknown augmentation {0}".format(augmentation))

    def GetFeatures(self, 
                    videoPath, 
                    frame_offset=0, 
                    frame_end_offset=0, 
                    retries=0, 
                    return_data=False, 
                    replicate_videoPaths=[], 
                    apply_augmentations=False,
                    predetermined_augmentation=None):
        """
        Return second level features for DeepFake classification.

        Parameters:
        videoPath (string): Path to .mp4 video.
        frame_offset (int): Start sampling from this frame. Default=0.
        frame_end_offset (int): Offset for end of video. This is automatically increased when faces are not detected from the last sampled frame
        retries (int): Number of retries done already. This is incremented when calling recursively on face detection errors until max_retries is reached. Default=0.
        apply_augmentations (bool): augmentation to face images. This is intended for training second level models, default=False.
        predetermined_augmentation (str) : One of fps_15, resize_smaller, jpeg_compression or None, default=None.

        Throws: Exception if max_retries is reached.
        """
        # Augmentations are None by default
        augmentation = predetermined_augmentation if not apply_augmentations else self._getRandomAugmentation()

        imgs, brightness_factor = self.frameSampler.getFrames(videoPath,
                                                              frame_offset=frame_offset,                     # read starting frame, this is increased on every retry  
                                                              frame_end_offset=frame_end_offset,
                                                              first_frame_step=2 if (augmentation == VideoAugmentation.HALF_FPS) else 1,
                                                              increase_lightness=retries>=2     # force brightness increase after second try
                                                              )
        
        replicate_imgs = [self.frameSampler.getFrames(path,
                                                      frame_offset,
                                                      frame_end_offset=frame_end_offset,
                                                      first_frame_step=2 if (augmentation == VideoAugmentation.HALF_FPS) else 1,
                                                      override_brightness_fc=brightness_factor  # use the same brightnesses in replicate videos as in original
                                                      )[0] for path in replicate_videoPaths]

        # apply augmentation - this is an identity function if augmentation is None
        imgs = self._augmentPILImages(imgs, augmentation)
        replicate_imgs = self._augmentPILImages(replicate_imgs, augmentation)
        min_image_side = min(imgs[0].width, imgs[0].height)
        
        face_dets_list, average_person_count, faceDetectionError = self.faceDetector.getFaceBoundingBoxes(imgs,
                                                                                      use_search_limits=retries<2 and min_image_side >= 480, # use search limits on the first two tries if the image has a decent resolution
                                                                                      speedOverAccuracy=retries==0 # try the first frame with only mtcnn model
                                                                                      )

        # retry if faces were not found from all frames
        if len(face_dets_list) < (self.n_first_frames + self.n_spaced_frames):
            if retries < self.max_retries:
                skip_frames_for_next_try = self.n_first_frames * 2 * (retries+1) # if first_frames=10, skips are 20, 20+40=60, 60+60=120, 120+80=200
                skip_end_frames_for_next_try = 10 if (faceDetectionError == FaceDetectorError.MISSING_LAST) else 0
                return self.GetFeatures(videoPath, 
                                        frame_offset = frame_offset + skip_frames_for_next_try, 
                                        frame_end_offset = frame_end_offset + skip_end_frames_for_next_try,
                                        retries = retries+1, 
                                        return_data = return_data, 
                                        replicate_videoPaths = replicate_videoPaths, 
                                        predetermined_augmentation=augmentation)
            else:
                raise Exception("Maximum retries with " + str(videoPath))

        # create tracked persons out of facedetections
        def __PILs2Numpys(pil_imgs):
            return [np.array(img, dtype=np.uint8) for img in pil_imgs]
        
        trackedPersons = extractPersons(face_dets_list,
                                        average_person_count=average_person_count, 
                                        np_imgs=__PILs2Numpys(imgs),
                                        small_face_size=self.small_face_size,
                                        large_face_size=self.large_face_size,
                                        face_padding=self.face_padding,
                                        n_first_images=self.n_first_frames,
                                        face_discard_percentage=self.face_discard_percentage)
        trackedReplicatePersons = [extractPersons(face_dets_list, average_person_count, __PILs2Numpys(imgs)) for imgs in replicate_imgs]

        # refine tracked person's faces to embeddings
        def __faces2embeddings(trackedPerson):
            trackedPerson.faceEmbeddings = self.faceEmbeddings.getEmbeddings(trackedPerson.small_faces_array)
        
        def __listOfFaces2embeddings(trackedPersonsList):
            for trackedPerson in trackedPersonsList:
                __faces2embeddings(trackedPerson)
        
        __listOfFaces2embeddings(trackedPersons)
        for replicates in trackedReplicatePersons:
            __listOfFaces2embeddings(replicates)

        # if return data instead of features
        if return_data:
            allPersons = trackedPersons
            for replicatePersons in trackedReplicatePersons:
                allPersons += replicatePersons
            return [person.getData() for person in allPersons]

        def __collectPersonFeatures(trackedPerson):
            return np.array(
                self.faceClassifier.getFaceClassifierFeats(trackedPerson.small_faces_array, isSmall=True, weights=trackedPerson.getWeights())+
                self.faceClassifier.getFaceClassifierFeats(trackedPerson.large_faces_array, isSmall=False, weights=trackedPerson.getWeights())+
                self.faceEmbeddings.getFaceEmbeddingFeatures(trackedPerson.faceEmbeddings)+
                self.faceSequenceClassifier.getFaceClassifierFeats(trackedPerson.large_faces_array) + 
                (self.powerSpectrumClassifier.getFeatures(trackedPerson.raw_faces_list) if self.use_power_spectrum_clf else [])
            )

        return [__collectPersonFeatures(trackedPerson) for trackedPerson in trackedPersons]

    def GetFeatureNames(self):
        names = []
        names += self.faceClassifier.getFeatNames()
        names += self.faceEmbeddings.getFeatNames()
        names += self.faceSequenceClassifier.getFeatNames()
        if self.use_power_spectrum_clf:
            names += self.powerSpectrumClassifier.getFeatNames()

        return np.array(names)

    def Predict(self, videoPath, frame_offset=0, 
                handleErrors=True, apply_augmentations=False, 
                featureClassifiers=['xgb','logreg','lightgbm'], multiPersonMode='max'):
        """ 
        Prediction for any fake persons in the video. Returns confidence for a fake [0-1]. 
        
        videoPath (str): Videofile path
        frame_offset (int): start processing video from this frame, default=0
        handleErrors (bool): If True (default), the method handles exceptions and outputs predict_on_error. If False, the exception is passed to caller.
        featureClassifiers (listo of str): Listo of what feature classifiers to combine. Available options are: xgb, logreg and lightgbm. All are included by default.
        multiPersonMode (str): How to combine predictions of multiple persons. One of max, avg, weighted-avg. In weighted avg, weights are 1 and 2 for <0.5 and >=0.5 predictions. Default=max
        """

        def __predict():
            # collect features from each person
            feats_list = self.GetFeatures(videoPath, frame_offset=frame_offset, apply_augmentations=apply_augmentations)
            person_preds = [self.secondLevelClassifier.predict(feats, featureClassifiers=featureClassifiers) for feats in feats_list]

            if self.verbose > 1:
                print("Person predictions: {0}".format(person_preds))
            # one of the persons can be fake so take max because 0=real and 1=fake 
            if multiPersonMode == 'max':
                return max(person_preds)
            if multiPersonMode == 'avg':
                return np.mean(np.array(person_preds))
            if multiPersonMode == 'weighted-avg':
                return np.average(np.array(person_preds),weights=np.where(np.array(person_preds) < 0.5,1,2))

        if(handleErrors):
            try:
                return __predict()
            except:
                print("Could not predict " + str(videoPath) + ". Predicting {0}.".format(self.predict_on_error))
                return self.predict_on_error
        else:
            # Allow to crash for debugging purposes or for external error handling
            return __predict()
            
        