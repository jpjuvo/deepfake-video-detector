import numpy as np

class FaceDetection:
    """
    Face detection class holds the face detection bounding boxes as well as detected landmarks.
    
    Class variables:
    ## Bounding box coordinates
    xmin: left side x-coordinate of the face bounding box
    ymin: top side y-coordinate of the face bounding box
    xmax: right side x-coordinate of the face bounding box
    ymax: bottom side y-coordinate of the face bounding box

    ## Detector model
    score: [0,1] confidence of the face detection. This comes from the model that predicted the face.

    ## Input image
    im_width: input frame width
    im_height: input frame height

    ## Landmarks - eyes, nose, mouth
    left_eye_x
    left_eye_y
    right_eye_x
    right_eye_y
    nose_x
    nose_y
    mouth_x
    mouth_y
    
    Class methods:
    bb(): returns bounding box numpy array - xmin,ymin,xmax,ymax
    width(): returns bounding box width
    height(): return bounding box height
    landmarks(): returns all landmark coordinates as numpy array
    isValid(): Checks whether bounding box coordinates are outside image area (or if detection is too small or too low)

    """
    def __init__(self,
                 xmin,ymin,xmax,ymax,score, im_width, im_height,
                 mtcnn_landmarks=None, blazeface_landmarks=None, landmarks=None):
        """
        xmin: left
        ymin: top
        xmax: right
        ymax: bottom
        score: confidence
        im_width: frame width
        im_height: frame height
        mtcnn_landmarks: list of x,y-coordinate lists of left eye,right eye, nose, left mouse corner, right mouse corner
        blazeface_landmarks: x,y-coordinates of right eye, left eye, nose, mouth, right ear, left ear
        landmarks: x,y-coordinates of left eye, right eye, nose, mouth
        """

        # Parse landmarks
        if mtcnn_landmarks is not None:
            # blazeface doesn't have mouth corners so calculate the center of mouth
            (left_eye,right_eye,nose,left_mouth,right_mouth) = mtcnn_landmarks
            self.left_eye_x,self.left_eye_y = left_eye
            self.right_eye_x,self.right_eye_y = right_eye
            self.nose_x,self.nose_y = nose
            left_mouth_x,left_mouth_y = left_mouth
            right_mouth_x,right_mouth_y = right_mouth
            # get center of the mouth from middle of the mouth corners
            self.mouth_x = (left_mouth_x + right_mouth_x)/2
            self.mouth_y = (left_mouth_y + right_mouth_y)/2
        elif blazeface_landmarks is not None:
            # ignore ear landmarks because these are missing from mtcnn
            (self.right_eye_x,self.right_eye_y,
            self.left_eye_x,self.left_eye_y,
            self.nose_x,self.nose_y,
            self.mouth_x,self.mouth_y,_,_,_,_) = blazeface_landmarks
        elif landmarks is not None:
            (self.right_eye_x,self.right_eye_y,
            self.left_eye_x,self.left_eye_y,
            self.nose_x,self.nose_y,
            self.mouth_x,self.mouth_y) = landmarks
        
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        self.score=score
        self.im_width = im_width
        self.im_height = im_height

    @classmethod
    def interpolateFromTwoDetections(cls, face_det_prev, face_det_next):
        """Initialize face detection by interpolating from previous and following detections"""
        bb = (face_det_prev.bb() + face_det_next.bb()) / 2
        score = 0.25 # not confident
        landmarks = (face_det_prev.raw_landmarks() + face_det_next.raw_landmarks()) / 2

        return cls(xmin=bb[0],
                   ymin=bb[1],
                   xmax=bb[2],
                   ymax=bb[3],
                   score=score, 
                   im_width=face_det_prev.im_width,
                   im_height=face_det_prev.im_height,
                   landmarks=landmarks)

    def bb(self):
        """Returns face bounding box coordinates as a numpy array"""
        return np.array([self.xmin,self.ymin,self.xmax,self.ymax])

    def width(self):
        """Returns face bounding box width"""
        return self.xmax-self.xmin

    def height(self):
        """Returns face bounding box height"""
        return self.ymax-self.ymin
    
    def raw_landmarks(self):
        return np.array([self.right_eye_x,
                         self.right_eye_y,
                         self.left_eye_x,
                         self.left_eye_y,
                         self.nose_x,
                         self.nose_y,
                         self.mouth_x,
                         self.mouth_y])

    def landmarks(self):
        """
        Returns facial landmarks x and y coordinates (numpy array). 
        The returned landmarks are left eye, right eye, nose and mouth.
        """
        return np.array([np.array([self.right_eye_x - self.xmin,
                         self.right_eye_y - self.ymin]),
                         np.array([self.left_eye_x - self.xmin,
                         self.left_eye_y - self.ymin]),
                         np.array([self.nose_x - self.xmin,
                         self.nose_y - self.ymin]),
                         np.array([self.mouth_x - self.xmin,
                         self.mouth_y - self.ymin])])

    def isValid(self,min_face_size=None, search_only_top_part=True, verbose=0):
        """
        Checks whether bounding box coordinates are outside image area (or if detection is too small or too low)

        returns (bool)
        """
        # Check for too small coordinates
        if self.ymin < 0 or self.xmin < 0:
            if verbose > 0:
                print("Discarding detection because outside frame area: {0}".format(self.bb()))
            return False
        # Check for inverse order in coordinates
        if self.ymax < self.ymin or self.xmax < self.xmin:
            if verbose > 0:
                print("Discarding detection because inverse order of coords: {0}".format(self.bb()))
            return False
        # Check for too large coordinates
        if self.ymax >= self.im_height or self.xmax >= self.im_width:
            if verbose > 0:
                print("Discarding detection because outside frame area: {0}".format(self.bb()))
            return False
        # Check for too small face
        if min_face_size is not None:
            if (self.width()) < min_face_size or (self.height())< min_face_size:
                if verbose > 0:
                    print("Discarding detection because too small: {0}".format(self.bb()))
                return False
        # Check for detection in the bottom third of the frame
        if search_only_top_part and ((self.ymax+self.ymin)//2 > (2*self.im_height//3)):
            return False
        return True