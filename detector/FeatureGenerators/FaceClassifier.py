import numpy as np
import cv2
import torch
import fastai
from fastai.vision import *
from Util.Timer import Timer
from Util.FeatureStats import preds2features, getStatFeatNames

class ArrayImageList(ImageList):
    """Custom Fastai ImageList that is constructed from a numpy image array."""

    @classmethod
    def from_numpy(cls, numpy_array):
        return cls(items=numpy_array)
    
    def label_from_array(self, array, label_cls=None, **kwargs):
        return self._label_from_list(array,label_cls=label_cls,**kwargs)
    
    def get(self, i):
        n = self.items[i]
        def _numpy2fastaiImage(img_arr):
            return fastai.vision.Image(pil2tensor(img_arr, dtype=np.float32).div_(255))
        return _numpy2fastaiImage(n)

class TTA:
    NONE = 'original'
    BRIGHT = 'bright'
    ZOOM = 'zoom'

class FaceClassifier:
    
    def __init__(self,
                 small_face_model_dirs,
                 large_face_model_dirs,
                 n_first_frames,
                 n_spaced_frames,
                 verbose=0):
        self.learn_small_faces = [load_learner(path=small_face_model_dir) for small_face_model_dir in small_face_model_dirs]
        self.learn_large_faces = [load_learner(path=large_face_model_dir) for large_face_model_dir in large_face_model_dirs]
        self.n_first_frames = n_first_frames
        self.n_spaced_frames = n_spaced_frames
        self.TTAs = [TTA.NONE, TTA.BRIGHT, TTA.ZOOM] # Zoom TTA is disabled to save inference time
        self.verbose = verbose
        print("Loaded {0} small face classifier and {1} large face classifier models.".format(self.size()[0],self.size()[1]))
        # print out paths
        for i,path in enumerate(small_face_model_dirs):
            print("{0} - Small face model: {1}".format(i,path))
        for i,path in enumerate(large_face_model_dirs):
            print("{0} - Large face model: {1}".format(i,path))

    def size(self):
        return (len(self.learn_small_faces),len(self.learn_large_faces))

    def getAugmentations(self, tta=TTA.NONE):
        if tta == TTA.NONE:
            return []
        elif tta == TTA.BRIGHT:
            return get_transforms(do_flip=True,
                                   flip_vert=False,
                                   max_rotate=0,
                                   max_zoom=1.0,
                                   max_lighting=0,
                                   max_warp=0.0,
                                   p_affine=1,
                                   p_lighting=1,
                                   xtra_tfms=[brightness(change=0.7)])
        elif tta == TTA.ZOOM:
            return get_transforms(do_flip=True,
                                   flip_vert=False,
                                   max_rotate=0,
                                   max_zoom=1.0,
                                   max_lighting=0,
                                   max_warp=0.0,
                                   p_affine=1,
                                   p_lighting=1,
                                   xtra_tfms=zoom_crop(scale=1.2))
        else:
            raise "Unrecognized TTA - {0}".format(tta)

    def getFeatNames(self):
        """Returns a list of feature names"""
        FF_names = []
        statFeatNames = getStatFeatNames()
        for size in ['small','large']:
            for tta in self.TTAs:
                for i in range(self.size()[0] if size=='small' else self.size()[1]):
                    for mode in ['first','spaced']:
                        FF_names += ["{0}_face_clf_{2}_{1}_{3}_{4}".format(size,i,statName, tta, mode) for statName in statFeatNames]
        return FF_names

    def getFaceClassifierFeats(self, faces_array, isSmall, weights=None):
        timer = Timer()
        n_total_frames = faces_array.shape[0]
        feats_list = []

        def isSoftmaxOutput(preds, eps=1e-6):
            mean = torch.mean(torch.sum(preds,dim=1))
            return (torch.abs(mean-1.0) < eps).item()

        for tta in self.TTAs:

            testDataSet = (ArrayImageList.from_numpy(faces_array)
                                        .split_none()
                                        .label_empty()
                                        .transform(self.getAugmentations(tta))
                                        .databunch(bs=n_total_frames)).normalize(imagenet_stats)
            testDataSet.train_dl = testDataSet.train_dl.new(shuffle=False)  # set shuffle off to kee the order

            test_batch = None

            for learn in self.learn_small_faces if isSmall else self.learn_large_faces:

                if test_batch is None:
                    # one batch is the whole test set
                    test_batch = learn.data.norm(testDataSet.one_batch())[0].cuda()

                # Get predictions and check if the model outputs softmax preds. If not, apply softmax.
                raw_preds = learn.pred_batch(ds_type=DatasetType.Test,
                                         batch=(test_batch,None))
                if not isSoftmaxOutput(raw_preds):
                    raw_preds = torch.softmax(raw_preds, dim=1)
                            
                # get Fake category preds from test set
                # The models are trained with 0:'FAKE' 1:'REAL' labels so first softmax index is the fake
                preds = raw_preds.numpy()[:,0]

                if self.verbose > 1:
                    print("Face classifier {0} preds: {1}".format("small" if isSmall else "large", preds))

                # get statistical features from the preds
                feats_list += preds2features(preds[:self.n_first_frames],
                                            weights=None if weights is None else weights[:self.n_first_frames],
                                            remove_n_outliers=0)
                feats_list += preds2features(preds[self.n_first_frames:],
                                            weights=None if weights is None else weights[self.n_first_frames:],
                                            remove_n_outliers=0)
                del raw_preds
        
            del test_batch
            del testDataSet

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)
        return feats_list
