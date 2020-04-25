import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm as tqdm

def getBalancedImageDataFrame(base_dir,df,n_frames=17, n_frames_start=0, avoid_sampling_classes=[], downsample_majority=True):
    
    # Start a list of paths, labels, and isVal splits
    paths_reals, labels_reals, isVals_reals = [],[],[]
    paths_fakes, labels_fakes, isVals_fakes = [],[],[]
    
    def _getImagePath(video_id,frame_index,person_index=0):
        name = video_id.replace('.mp4','_{0}_{1}.png'.format(person_index,frame_index))
        return str(os.path.join(base_dir,name))
    
    def _getFrameLabel(df, df_index, frame_index, person_index=0):
        # get array of frame labels from the dataframe
        frame_lbl_col = 'first_person_frame_labels' if person_index==0 else 'second_person_frame_labels'
        frame_labels = df.iloc[df_index][frame_lbl_col]
        
        # check for nan
        if frame_labels == np.nan:
            return np.nan
        
        lbls_int = [int(lbl) for lbl in str(frame_labels).replace('[','').replace(']','').replace(' ','').split(',')]
        return lbls_int[frame_index]
    
    def _combineTwoListsAlternating(list1,list2):
        combined = [None]*(len(list1)+len(list2))
        combined[::2] = list1
        combined[1::2] = list2
        return combined
    
    # Get all real videos and go through one by one
    df_reals = df[df['label']=='REAL']
    df_fakes = df[df['label']=='FAKE']
    
    # shuffle reals
    real_indices = list(df_reals.index.values)
    random.shuffle(real_indices)
    
    for real_ind in tqdm(real_indices):
        real_id = df_reals.loc[real_ind,'index']
        isVal = df_reals.loc[real_ind,'isValFold']
    
        # Get all fake replicates for the real
        fake_replicates = df_fakes[df_fakes['original']==real_id]
        if len(fake_replicates)==0:
            continue

        # Get all frames for the real and check that they exist.
        for frame_index in range(n_frames_start,n_frames):
            
            real_frame = _getImagePath(real_id,frame_index)
            if not os.path.isfile(real_frame):
                continue
            
            # For each frame in real, sample a fake pair randomly and check a) that it exists b) it's not zero class
            # If not exist or on every second two-class, take the next one.
            fake_index = 0
            fake_label = 0
            fake_path = ""
            loop_counter = len(fake_replicates)
            while loop_counter > 0:
                loop_counter -= 1
                # random pick fake one - if oversampling reals, pick 
                fake_index = random.randint(0,len(fake_replicates)-1)
                fake_label = _getFrameLabel(fake_replicates, fake_index, frame_index)
                
                # check that the fake image exists
                fake_path = _getImagePath(fake_replicates.iloc[fake_index]['index'], frame_index)
                if not os.path.isfile(fake_path):
                    continue
                
                # skip the majority class randomly half of the time
                skip_majority_class = random.randint(0,1)==0 and downsample_majority

                # is the sampling condition satisfied? Avoid sampling unaltered fakes or too many majority types.
                # avoid_sampling_classes avoids classes except for validation fold 
                if (fake_label != 0 and (fake_label not in avoid_sampling_classes or isVal) and (skip_majority_class and fake_label != 2)):
                    break
            
            # Check that fake sample got selected
            if not os.path.isfile(fake_path):
                continue
            # Check that we didn't end up with 0 labeled fake
            if fake_label == 0:
                continue
            
            if (fake_label in avoid_sampling_classes and not isVal):
                continue

            # Append lists
            paths_reals.append(real_frame)
            labels_reals.append(int(0))
            isVals_reals.append(isVal)
            
            paths_fakes.append(fake_path)
            labels_fakes.append(int(fake_label))
            isVals_fakes.append(fake_replicates.iloc[fake_index]['isValFold'])
    
    # shuffle lists but maintain the same order between lists
    zipped_list = list(zip(paths_reals,labels_reals,isVals_reals,paths_fakes,labels_fakes,isVals_fakes))
    random.shuffle(zipped_list)
    paths_reals,labels_reals,isVals_reals,paths_fakes,labels_fakes,isVals_fakes = zip(*zipped_list)
    
    # combine real and fake lists in an alternating fashion so that the order is real1,fake1,real2,fake2,...
    paths = _combineTwoListsAlternating(paths_reals,paths_fakes)
    labels = _combineTwoListsAlternating(labels_reals,labels_fakes)
    isVals = _combineTwoListsAlternating(isVals_reals,isVals_fakes)
    
    return pd.DataFrame({'path':paths,'label':labels, 'isValFold':isVals})


def getBalancedVideoDataFrame(df):
    
    # Start a list of paths, labels, and isVal splits
    paths_reals, labels_reals, isVals_reals = [],[],[]
    paths_fakes, labels_fakes, isVals_fakes = [],[],[]
    
    def _getVideoLabel(df, df_index, person_index=0):
        # get array of frame labels from the dataframe
        frame_lbl_col = 'first_person_label' if person_index==0 else 'second_person_label'
        lbl = df.iloc[df_index][frame_lbl_col]
        
        # check for nan
        if lbl == np.nan:
            return np.nan

        return int(lbl)
    
    def _combineTwoListsAlternating(list1,list2):
        combined = [None]*(len(list1)+len(list2))
        combined[::2] = list1
        combined[1::2] = list2
        return combined
    
    # Get all real videos and go through one by one
    df_reals = df[df['label']=='REAL']
    df_fakes = df[df['label']=='FAKE']
    
    # shuffle reals
    real_indices = list(df_reals.index.values)
    random.shuffle(real_indices)
    
    for real_ind in tqdm(real_indices):
        real_id = df_reals.loc[real_ind,'index']
    
        # Get all fake replicates for the real
        fake_replicates = df_fakes[df_fakes['original']==real_id]
        if len(fake_replicates)==0:
            continue
            
        # sample a fake pair randomly and check a) it's not zero class
        # on every second two-class, take the next one.
        fake_index = -1
        fake_label = 0
        loop_counter = len(fake_replicates)
        while loop_counter > 0:
            loop_counter -= 1
            # random pick fake one - if oversampling reals, pick 
            fake_index = random.randint(0,len(fake_replicates)-1)
            fake_label = _getVideoLabel(fake_replicates, fake_index)
            
            # skip the majority class randomly half of the time
            skip_majority_class = random.randint(0,1)==0
            
            # is the sampling condition satisfied? Avoid sampling unaltered fakes or too many majority types.
            if (fake_label != 0 and (skip_majority_class and fake_label != 2)): 
                break
            
        if fake_index == -1:
            continue

        # Append lists
        paths_reals.append(real_id)
        labels_reals.append(int(0))
        isVals_reals.append(df_reals.loc[real_ind,'isValFold'])
        
        paths_fakes.append(fake_replicates.iloc[fake_index]['index'])
        labels_fakes.append(int(fake_label))
        isVals_fakes.append(fake_replicates.iloc[fake_index]['isValFold'])
    
    # shuffle lists but maintain the same order between lists
    zipped_list = list(zip(paths_reals,labels_reals,isVals_reals,paths_fakes,labels_fakes,isVals_fakes))
    random.shuffle(zipped_list)
    paths_reals,labels_reals,isVals_reals,paths_fakes,labels_fakes,isVals_fakes = zip(*zipped_list)
    
    # combine real and fake lists in an alternating fashion so that the order is real1,fake1,real2,fake2,...
    paths = _combineTwoListsAlternating(paths_reals,paths_fakes)
    labels = _combineTwoListsAlternating(labels_reals,labels_fakes)
    isVals = _combineTwoListsAlternating(isVals_reals,isVals_fakes)
    
    return pd.DataFrame({'path':paths,'label':labels, 'isValFold':isVals})

def getAllImagesDataFrame(base_dir,df,n_frames=17):
    
    # Start a list of paths, labels, and isVal splits
    paths, labels, isVals = [],[],[]
    
    def _getImagePath(video_id,frame_index,person_index=0):
        name = video_id.replace('.mp4','_{0}_{1}.png'.format(person_index,frame_index))
        return str(os.path.join(base_dir,name))
    
    def _getFrameLabel(df, df_index, frame_index, person_index=0):
        # get array of frame labels from the dataframe
        frame_lbl_col = 'first_person_frame_labels' if person_index==0 else 'second_person_frame_labels'
        frame_labels = df.iloc[df_index][frame_lbl_col]
        
        # check for nan
        if frame_labels == np.nan:
            return np.nan
        
        lbls_int = [int(lbl) for lbl in str(frame_labels).replace('[','').replace(']','').replace(' ','').split(',')]
        return lbls_int[frame_index]
    
    # Get all real videos and go through one by one
    df_reals = df[df['label']=='REAL']
    df_fakes = df[df['label']=='FAKE']
    
    # shuffle reals
    real_indices = list(df_reals.index.values)
    random.shuffle(real_indices)
    
    for real_ind in tqdm(real_indices):
        real_id = df_reals.loc[real_ind,'index']
        isVal = df_reals.loc[real_ind,'isValFold']
    
        # Get all fake replicates for the real
        fake_replicates = df_fakes[df_fakes['original']==real_id]
        if len(fake_replicates)==0:
            continue

        # Get all frames for the real and check that they exist.
        for frame_index in range(n_frames):
            
            real_frame = _getImagePath(real_id,frame_index)
            if not os.path.isfile(real_frame):
                continue

            # Append lists
            paths.append(real_frame)
            labels.append(int(0))
            isVals.append(isVal)
            
            fake_index = 0
            fake_label = 0
            fake_path = ""
            loop_counter = len(fake_replicates)
            while loop_counter > 0:
                loop_counter -= 1
                # random pick fake one - if oversampling reals, pick 
                fake_index = loop_counter-1
                fake_label = _getFrameLabel(fake_replicates, fake_index, frame_index)
                
                # check that the fake image exists
                fake_path = _getImagePath(fake_replicates.iloc[fake_index]['index'], frame_index)
                if not os.path.isfile(fake_path):
                    continue
                
                # Append lists
                paths.append(fake_path)
                labels.append(fake_label)
                isVals.append(fake_replicates.iloc[fake_index]['isValFold'])
    
    return pd.DataFrame({'path':paths,'label':labels, 'isValFold':isVals})

    