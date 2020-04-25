import numpy as np
import PIL
import cv2

def JPEGCompression(img, compression=70):
    """ JPEG encode PIL image """
    np_img = np.array(img, dtype=np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression]
    _, encimg = cv2.imencode('.jpg', np_img, encode_param)
    return PIL.Image.fromarray(cv2.imdecode(encimg, 1))

def ResizeImage(img, scale=0.25):
    """ Resize PIL image """
    width = int(img.width * scale)
    height = int(img.height * scale)
    dim = (width, height)
    return img.resize(dim, resample=PIL.Image.BILINEAR)

