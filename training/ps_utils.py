# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
import numpy as np
import cv2

EPSILON = 1e-8 # to prevent infinities

def cropOrPadIfNeeded(img, tgt_width=450, tgt_height=450, border_color=0):
    ''' '''
    
    h,w = img.shape[0:2]
    x = w-tgt_width
    x_left = int(np.ceil(abs(x)/2))
    x_right = int(np.floor(abs(x)/2))
    
    y = h-tgt_height
    y_top = int(np.ceil(abs(y)/2))
    y_bottom = int(np.floor(abs(y)/2))
    
    #images that need padding
    if x<0: 
        img = cv2.copyMakeBorder(img, 0, 0, x_left, x_right, border_color)
    
    if y<0: 
        img = cv2.copyMakeBorder(img, y_top, y_bottom, 0, 0, border_color)
    
    #images that need cropping
    if y>0: 
        img = img[y_top:h-y_bottom]
    if x>0: 
        img = img[:, x_left:w-x_right]

    return img

def azimuthalAverage(image, center=None):
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

def makeSquare(im, size:int=None, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, color = [0, 0, 0]):
    '''
    '''
    
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


def imgTo1DPS(img:np.array, tgt_height=100, tgt_width=100, border_color=0):
    '''Process an image (in BGR mode) to 1D Power Spectrum'''
    
    assert len(img.shape)==3 and img.shape[-1]==3
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cropOrPadIfNeeded(img, tgt_height=tgt_height, tgt_width=tgt_width, border_color=border_color)

    
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift)+EPSILON)
    

    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(magnitude_spectrum)
    if any([not np.isfinite(k) for k in psd1D]):
        raise ValueError("Bad values occured")
    
    psd1D /= psd1D[0]
    
    return psd1D
