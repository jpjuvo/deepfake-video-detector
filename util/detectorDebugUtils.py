import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys

def plotFaceSamples(datasOfTrackedPersons, n_frames=17, title=None):

    f,ax = plt.subplots(len(datasOfTrackedPersons) * 3,n_frames, figsize=(15,len(datasOfTrackedPersons)*4))
    for personInd, personDatas in enumerate(datasOfTrackedPersons):
        
        _,small_faces, large_faces, raw_faces, landmarks, _, weights = personDatas

        if(len(small_faces) >= 1):
            for i in range(n_frames):
                ax[0 + personInd*3,i].imshow(small_faces[i])
                ax[0 + personInd*3,i].get_xaxis().set_visible(False)
                ax[0 + personInd*3,i].get_yaxis().set_visible(False)

                ax[1 + personInd*3,i].imshow(large_faces[i])
                ax[1 + personInd*3,i].get_xaxis().set_visible(False)
                ax[1 + personInd*3,i].get_yaxis().set_visible(False)

                ax[2 + personInd*3,i].imshow(raw_faces[i])
                ax[2 + personInd*3,i].get_xaxis().set_visible(False)
                ax[2 + personInd*3,i].get_yaxis().set_visible(False)
                #ax[2 + personInd*3,i].set_title('{0:.2f}'.format(weights[i]))

                lms = landmarks[i]
                ax[2 + personInd*3,i].scatter(x=lms[:,0], y=lms[:,1], c='r', s=2)

    if title is not None:
        f.suptitle(title, fontsize=18)

    plt.show()

def analyzePred(deepFakeDetector, videopath, real_label, n_frames=20):
    pred = deepFakeDetector.Predict(videopath)
    data = deepFakeDetector.GetFeatures(videopath, return_data=True)
    plotFaceSamples(data, n_frames=n_frames, title='Prediction {0:.2f} - GT {1}'.format(pred, real_label))

def randomColors(N, bright=True, shuffle=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        np.random.shuffle(colors)
    colors=colors*np.array([255, 255, 255])
    colors = colors.astype(np.uint8)
    return colors

def showLandmarkTrack(datasOfTrackedPersons, frameStart=0, frameEnd=16):
    f,ax = plt.subplots(1, len(datasOfTrackedPersons), figsize=(8,15))
    
    colors = randomColors(4, shuffle=False)
    for i, personDatas in enumerate(datasOfTrackedPersons):
        _,_,_,raw_faces, landmarks, landmark_samples, _ = personDatas
        img = np.zeros(raw_faces[0].shape,np.uint8)
        
        p0s = None
        for p1s,color_samples in zip(landmarks[frameStart:frameEnd], landmark_samples[frameStart:frameEnd]):
            if p0s is None:
                p0s = p1s
                continue
            left_eye_hue = color_samples[0][0]
            right_eye_hue = color_samples[1][0]
            eye_hue_symmetry_distance = abs(left_eye_hue-right_eye_hue)
            if eye_hue_symmetry_distance > 0.5:
                eye_hue_symmetry_distance = 1-eye_hue_symmetry_distance
            print("Eye hue symmetry distance {0:.3f}".format(eye_hue_symmetry_distance))
            for j, (_,color_sample) in enumerate(zip(p1s,color_samples)):
                #color = (int(colors[j][0]), int(colors[j][1]), int(colors[j][2]))
                rgb = colorsys.hsv_to_rgb(*color_sample)
                color = (int(255*rgb[0]), int(255*rgb[1]), int(255*rgb[2]))
                cv2.line(img, (int(p0s[j][0]),int(p0s[j][1])), 
                         (int(p1s[j][0]),int(p1s[j][1])), color, 1)
            p0s = p1s
        if(len(datasOfTrackedPersons)>1):
            ax[i].imshow(img)
        else:
            ax.imshow(img)
    plt.show()