import numpy as np
import os
import cv2
import torch
import fastai
from fastai.vision import *
from Util.Timer import Timer
from Util.FeatureStats import preds2features, getStatFeatNames

from RecurrentModel.RecurrentCNN import MixedVideoSequenceModel, VideoSequenceModel
from RecurrentModel.RecurrentModelConfig import RecurrentModelConfig
from RecurrentModel.ImageSequence import ImageSequence
from RecurrentModel.ImageSequenceList import ImageSequenceList

class ArrayImageSequenceList(ImageList):
    """Custom Fastai ImageList that is constructed from a numpy image array."""

    @classmethod
    def from_numpy(cls, numpy_array):
        return cls(items=numpy_array)
    
    def label_from_array(self, array, label_cls=None, **kwargs):
        return self._label_from_list(array,label_cls=label_cls,**kwargs)
    
    def get(self, i):
        n = self.items[i]
        return ImageSequence(list(n))

class TTA:
    NONE = 'original'
    BRIGHT = 'bright'
    ZOOM = 'zoom'

class FaceSequenceClassifier:
    
    def __init__(self,
                 sequence_model_dirs,
                 n_first_frames,
                 n_spaced_frames,
                 verbose=0):
        
        self.learn_sequence_models = [load_learner(path=model_dir) for model_dir in sequence_model_dirs]
        self.configs = [RecurrentModelConfig.fromDir(model_dir) for model_dir in sequence_model_dirs]
        
        self.n_first_frames = n_first_frames
        self.n_spaced_frames = n_spaced_frames
        self.TTAs = [TTA.NONE, TTA.BRIGHT, TTA.ZOOM]
        
        # number of sequence batches is low (1-3 batches)
        # this tells the statistics function not return std and median
        self.low_sample_count = False #True
        self.verbose = verbose
        print("Loaded {0} face sequence classifier models.".format(self.size()))

    def size(self):
        return len(self.learn_sequence_models)

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
        statFeatNames = getStatFeatNames(low_sample_count=self.low_sample_count)
        for tta in self.TTAs:
            for i in range(self.size()):
                FF_names += ["seq_clf_len-{0}_start-{1}_{2}_{3}_{4}".format(self.configs[i].getLenSequence(),
                                                                            self.configs[i].getMinFrameIndex(),
                                                                            i,
                                                                            tta,
                                                                            statName) for statName in statFeatNames]
        return FF_names

    def getFaceClassifierFeats(self, faces_array):
        timer = Timer()
        n_total_frames = faces_array.shape[0]
        feats_list = []

        def isSoftmaxOutput(preds, eps=1e-6):
            mean = torch.mean(torch.sum(preds,dim=1))
            return (torch.abs(mean-1.0) < eps).item()

        for learn, config in zip(self.learn_sequence_models, self.configs):
                
            len_seq = config.getLenSequence()
            min_frame = config.getMinFrameIndex()
            max_frame = config.getMaxFrameIndex()
            
            # there can be more samples than defined in config
            if max_frame < n_total_frames - 1 and min_frame >= self.n_first_frames:
                max_frame = n_total_frames - 1
            total_frames = (max_frame - min_frame)
            step_size = len_seq // 2
            bs = 1 + ((total_frames - len_seq) // step_size)
            batch = np.zeros((bs,config.getLenSequence(),*faces_array[0].shape),np.float32)

            for i in range(bs):
                    start = min_frame + i * step_size
                    batch[i] = faces_array[start:start+len_seq]

            for tta in self.TTAs:

                testDataSet = (ArrayImageSequenceList.from_numpy(batch)
                                                     .split_none()
                                                     .label_empty()
                                                     .transform(self.getAugmentations(tta))
                                                     .databunch(bs=bs))
                testDataSet.train_dl = testDataSet.train_dl.new(shuffle=False)  # set shuffle off to kee the order

                test_batch = None

                if test_batch is None:
                    # one batch is the whole test set
                    test_batch = testDataSet.one_batch()[0].cuda()

                # Get predictions and check if the model outputs softmax preds. If not, apply softmax.
                raw_preds = learn.pred_batch(ds_type=DatasetType.Test,
                                         batch=(test_batch,None))
                if not isSoftmaxOutput(raw_preds):
                    raw_preds = torch.softmax(raw_preds, dim=1)

                # get Fake category preds from test set
                # The models are trained with 0:'FAKE' 1:'REAL' labels so first softmax index is the fake
                preds = raw_preds.numpy()[:,0]

                if self.verbose > 1:
                    print("Face sequence clf preds: {0}".format(preds))

                # get statistical features from the preds
                feats_list += preds2features(preds,
                                             remove_n_outliers=0,
                                             low_sample_count=self.low_sample_count)
                del raw_preds
                del test_batch
                del testDataSet
            del batch

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)
        return feats_list
