import numpy as np
import cv2
import xgboost as xgb
from Util.Timer import Timer
from Util.FeatureStats import preds2features, getStatFeatNames

class PowerSpectrumClassifier:

    def __init__(self,
                 model_path,
                 verbose=0):
        self.epsilon = 1e-8
        self.MAX_SIZE = 600 # maximum image size
        self.BORDER_MODE = cv2.BORDER_REFLECT
        self.INTERPOLATION = cv2.INTER_NEAREST
        self.xgb_model = xgb.XGBClassifier()
        self.initialized = False
        self.verbose=verbose
        if model_path is not None:
            self.xgb_model.load_model(model_path)
            self.initialized = True
            print("Loaded power spectrum path")

    def _azimuthalAverage(self, image, center=None):
        """
        Calculate the azimuthally averaged radial profile.

        image - The 2D image
        center - The [x,y] pixel coordinates used as the center. The default is 
                None, which then uses the center of the image (including 
                fracitonal pixels).
        
        """
        # Calculate the indices from the image
        y, x = np.indices(image.shape)

        if not center:
            center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

        r = np.hypot(x - center[0], y - center[1])

        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]

        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)

        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]       # location of changed radius
        nr = rind[1:] - rind[:-1]        # number of radius bin
        
        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]

        radial_prof = tbin / nr

        return radial_prof

    def _imgTo1DPS(self, img:np.array, size:int=450, border_mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_NEAREST):
        '''Process an image (in BGR mode) to 1D Power Spectrum'''
        
        assert len(img.shape)==3 and img.shape[-1]==3
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self._makeSquare(img, size=size, border_mode=border_mode,  interpolation=interpolation)
       
        # Calculate FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift)+self.epsilon)
        
        # Calculate the azimuthally averaged 1D power spectrum
        psd1D = self._azimuthalAverage(magnitude_spectrum)
        if any([not np.isfinite(k) for k in psd1D]):
            raise ValueError("Bad values occured")
        
        psd1D /= psd1D[0]
        
        return psd1D


    def _makeSquare(self, im, size:int=None, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, color = [0, 0, 0]):
        ## Params
        old_size = im.shape[:2]
        if size:
            ratio = float(size)/max(old_size)
            new_size = tuple([int(x*ratio) for x in old_size])

            ## Resize
            im = cv2.resize(im, (new_size[1], new_size[0]), interpolation=interpolation) 
        
        else:
            size = max(old_size)
            new_size = old_size
            
        ## Count the required sizes for padding
        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        return cv2.copyMakeBorder(im, top, bottom, left, right, border_mode, value=color)

    def getFeatNames(self):
        statFeatNames = getStatFeatNames()
        FF_names = ['power_spectrum_{0}'.format(statName) for statName in statFeatNames]
        return FF_names

    def getFeatures(self, faces_array):
        timer = Timer()
        if not self.initialized:
            return list(np.ones(5)*0.5)
        
        preds = []
        for img in faces_array:
            psd1D = self._imgTo1DPS(img, size = self.MAX_SIZE, border_mode=self.BORDER_MODE, interpolation=self.INTERPOLATION)
            preds.append(self.xgb_model.predict_proba(psd1D[np.newaxis,...])[0][1])

        feats = preds2features(np.array(preds), remove_n_outliers=0)

        timer.print_elapsed(self.__class__.__name__, verbose=self.verbose)
        return feats
