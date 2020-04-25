import numpy as np
from PIL import Image
import cv2
import torch
from FaceDetection.BlazeFace import BlazeFace 
from facenet_pytorch import MTCNN
from FaceDetection.FaceDetection import FaceDetection
from Util.Timer import Timer
import math

class FaceDetectorError:
    NONE = ''
    MISSING_CONSECUTIVES = 'too many consecutive frames are missing'
    MISSING_FIRST = 'one or more frames are missing from the start'
    MISSING_LAST = 'one or more frames are missing from the end'
    ZERO_DETECTIONS = 'not a single face detected'
    TOO_FEW_DETECTIONS = 'less than three faces detected in all sample frames'

class FaceDetector:
    """Detects face bounding boxes from images"""
    def __init__(self,
                 device,
                 blazeface_device,
                 blazeface_path,
                 blazeface_anchors,
                 mtcnn_downsampling,
                 min_face_size=60,
                 verbose=0
                 ):
        self.downsampling = mtcnn_downsampling
        self.min_face_size = min_face_size
        self.mtcnn = MTCNN(keep_all=True, 
                           select_largest=False,
                           min_face_size=10,# init with very small size and filter them later
                           device=device).eval()
        self.blazeFaceUtil = BlazeFaceUtil(blazeface_device,blazeface_path,blazeface_anchors, min_face_size, verbose=verbose)
        self.frame_height = 1920
        self.frame_width = 1920
        self.verbose = verbose
        print("Loaded face detectors.")

    def _interpolateBBs(self, face_det_list):
        for i in range(1,len(face_det_list) - 1):
            if face_det_list[i] is None:
                face_det_prev = face_det_list[i-1][0]
                face_det_next = face_det_list[i+1][0]
                new_face_det = FaceDetection.interpolateFromTwoDetections(face_det_prev=face_det_prev, 
                                                                          face_det_next=face_det_next)

                face_det_list[i] = [new_face_det]
        return face_det_list

    def _getMTCNNFaceDetections(self, imgs, use_search_limits):
        full_size = imgs[0].size
        self.frame_height = imgs[0].height
        self.frame_width = imgs[0].width
        downsampling = self.downsampling if use_search_limits else 1

        # downsample images for faster detection
        ds_imgs = [img.resize((full_size[0]// downsampling, full_size[1]// downsampling)) for img in imgs]

        # crop bottom part away
        if use_search_limits:
            ds_imgs = [img.crop((0,0,img.width, 2*img.height//3)) for img in ds_imgs]

        face_detections_from_frames = []
        try: # mtcnn can throw
            face_dets_list, scores_list, landmarks_list = self.mtcnn.detect(ds_imgs,landmarks=True)
            # upsample detections back to the original size
            face_dets_list = [(face_det * downsampling) if face_det is not None else None for face_det in face_dets_list]
            landmarks_list = [(landmarks * downsampling) if landmarks is not None else None for landmarks in landmarks_list]
            
            # construct face detections
            for (facedets,scores,landmarks) in zip(face_dets_list,scores_list,landmarks_list):
                
                # continue if this frame is missing
                if facedets is None:
                    face_detections_from_frames.append(None)
                    continue

                face_detections = []
                for (bb,score,keypoints) in zip(facedets,scores,landmarks):
                    xmin, ymin, xmax, ymax = bb
                    
                    # clip values
                    xmin=np.clip(xmin,0,self.frame_width-1)
                    xmax=np.clip(xmax,0,self.frame_width-1)
                    ymin=np.clip(ymin,0,self.frame_height-1)
                    ymax=np.clip(ymax,0,self.frame_height-1)
                    
                    faceDet = FaceDetection(xmin,ymin,xmax,ymax,score,
                                            im_width=self.frame_width, 
                                            im_height=self.frame_height,
                                            mtcnn_landmarks=keypoints)
                    if not faceDet.isValid(min_face_size=self.min_face_size if use_search_limits else None,
                                        search_only_top_part=use_search_limits, verbose=self.verbose):
                        continue
                    face_detections.append(faceDet)

                face_detections_from_frames.append(face_detections if len(face_detections) != 0 else None)
        except:
            return []
        return face_detections_from_frames

    def _nms(self, facedets, n_max=2, increase_double_det_confidence=True):
        """Filter out the less confident overlapping face detections 
        and discard the least confident extra detections (>n_max)"""
        # don't do anything with one or less detection
        if(len(facedets)<=1):
            return facedets

        remove_list = []
        #remove overlapping facedetections
        for i in range(len(facedets)):
            for j in range(i + 1,len(facedets)):
                # detections are ordered the largest first
                # get leftmost, rightmost
                LM,RM = (i,j) if (facedets[i].xmin < facedets[j].xmin) else (j,i)
                if (facedets[RM].xmin < facedets[LM].xmax): # overlapping horizontally
                    # get topmost bottommost
                    TM, BM = (i,j) if (facedets[i].ymin < facedets[j].ymin) else (j,i)
                    if (facedets[BM].ymin < facedets[TM].ymax): # overlapping vertically
                        remove_list.append(j) # use the original order to judge importance
                        if increase_double_det_confidence:
                            # this is a confident detection - take root of average as the confidence score
                            average_score = (facedets[i].score + facedets[j].score)/2
                            facedets[i].score = math.sqrt(average_score) 
                        
        # return filtered and only n_max persons at maximum 
        keep_facedets = [facedets[i] for i in range(len(facedets)) if i not in remove_list]
        return keep_facedets[:min(len(keep_facedets),n_max)]

    def returnFaceDetectionError(self, error):
        returned_on_error = ([],1,error)
        if self.verbose > 0:
                print("Facedetector error - {0}".format(error))
        return returned_on_error

    def getFaceBoundingBoxes(self, imgs, use_search_limits=True, speedOverAccuracy=False):
        """
        Returns list of face bounding box lists and list of score lists.
        use_search_limits limits the face size to min_face_size and discards detections from the bottom third of the frame.

        MTCNN and Blazeface detections are both returned is speedOverAccuracy is False, else, only MTCNN is used. 
        MTCNN detections are first on the list.
        """
        
        # Run and time face detectors
        timer = Timer()
        face_dets_list_mtcnn = self._getMTCNNFaceDetections(imgs, use_search_limits=use_search_limits)
        timer.print_elapsed("MTCNN facedetector", verbose=self.verbose)

        face_dets_list_blazeface = [None]*len(imgs)
        if not speedOverAccuracy or any(x is None for x in face_dets_list_mtcnn):
            face_dets_list_blazeface = self.blazeFaceUtil.GetFaces(imgs, use_search_limits=use_search_limits)
            timer.print_elapsed("Blazeface facedetector", verbose=self.verbose)

        # concatenate face detections
        face_dets_list = []
        # go through frames
        for frame_index in range(len(imgs)):
            frame_detections = []
            mtcnn_detections = face_dets_list_mtcnn[frame_index] if len(face_dets_list_mtcnn) > frame_index else None
            blazeface_detections = face_dets_list_blazeface[frame_index] if len(face_dets_list_blazeface) > frame_index else None
            
            # include frame detections from both of the detector if they aren't empty
            if mtcnn_detections is not None:
                frame_detections += mtcnn_detections
            if blazeface_detections is not None:
                frame_detections += blazeface_detections
            face_dets_list.append(frame_detections if len(frame_detections) != 0 else None)

        # error case
        if len(face_dets_list) == 0:
            return self.returnFaceDetectionError(FaceDetectorError.ZERO_DETECTIONS)
        
        # Interpolate if single frames are missing a detection
        if any(x is None for x in face_dets_list):

            # Is it possible to interpolate? Extrapolation is not supported.
            firstFrameMissing = face_dets_list[0] is None
            lastFrameMissing = face_dets_list[-1] is None

            n_consecutivesMissing = 0
            tmp_consecutives_missing = 0
            for i in range(len(face_dets_list)):
                tmp_consecutives_missing = tmp_consecutives_missing + 1 if face_dets_list[i] is None else 0
                n_consecutivesMissing = max(n_consecutivesMissing, tmp_consecutives_missing)
            if lastFrameMissing:
                return self.returnFaceDetectionError(FaceDetectorError.MISSING_LAST)
            if firstFrameMissing:
                return self.returnFaceDetectionError(FaceDetectorError.MISSING_FIRST)
            if len(face_dets_list) < 3 :
                return self.returnFaceDetectionError(FaceDetectorError.TOO_FEW_DETECTIONS)
            if n_consecutivesMissing > 1:
                return self.returnFaceDetectionError(FaceDetectorError.MISSING_CONSECUTIVES)
            
            # fill missing bbs
            face_dets_list = self._interpolateBBs(face_dets_list)

        # NMS the combined facedetector preds
        average_person_count = 0
        for i in range(len(face_dets_list)):
            face_dets_list[i] = self._nms(face_dets_list[i])
            average_person_count += len(face_dets_list[i]) if face_dets_list[i] is not None else 0
        average_person_count = int(round(average_person_count/len(face_dets_list)))
        average_person_count = min(max(average_person_count,1),2)
        
        timer.print_elapsed("Face detection postprocess", verbose=self.verbose)
        return face_dets_list, average_person_count, FaceDetectorError.NONE

class BlazeFaceUtil:
    """
    This class is modified from Human Analog's code https://www.kaggle.com/humananalog/inference-demo
    """

    def __init__(self,
                 device,
                 blazeface_path,
                 blazeface_anchors,
                 min_face_size,
                 verbose=0):
        self.blazeface = BlazeFace().to(device)
        self.blazeface.load_weights(blazeface_path)
        self.blazeface.load_anchors(blazeface_anchors)
        self.min_face_size = min_face_size
        self.use_search_limits = True
        self.frame_height = 1920
        self.frame_width = 1920
        self.verbose=verbose
        _ = self.blazeface.train(False)

    def GetFaces(self, imgs, use_search_limits=True):
        self.use_search_limits = use_search_limits
        target_size = self.blazeface.input_size
        frames = np.array([np.array(img,dtype=np.uint8) for img in imgs])
        my_tiles, resize_info = self._tile_frames(frames, target_size)
        tiles=[my_tiles]

        # Put all the tiles for all the frames from all the videos into
        # a single batch.
        batch = np.concatenate(tiles)

        # Run the face detector. The result is a list of PyTorch tensors, 
        # one for each image in the batch.
        all_detections = self.blazeface.predict_on_batch(batch, apply_nms=False)
        # Convert the detections from 128x128 back to the original frame size.
        detections = self._resize_detections(all_detections, target_size, resize_info)

        # Because we have several tiles for each frame, combine the predictions
        # from these tiles. The result is a list of PyTorch tensors, but now one
        # for each frame (rather than each tile).
        num_frames = frames.shape[0]
        frame_size = (imgs[0].width, imgs[0].height)
        self.frame_height = imgs[0].height
        self.frame_width = imgs[0].width
        detections = self._untile_detections(num_frames, frame_size, detections)

        # The same face may have been detected in multiple tiles, so filter out
        # overlapping detections. This is done separately for each frame.
        detections = self.blazeface.nms(detections)

        face_det_list = []
        for i in range(len(detections)):
                # Crop the faces out of the original frame.
                faces = self._add_margin_to_detections(detections[i], frame_size, 0.2)
                faces_detections = self._get_face_detections(faces)
                face_det_list.append(faces_detections if len(faces_detections)>0 else None) # append None if missing to replicate MTCNN behaviour

        return face_det_list

    def _tile_frames(self, frames, target_size):
        """Splits each frame into several smaller, partially overlapping tiles
        and resizes each tile to target_size.

        After a bunch of experimentation, I found that for a 1920x1080 video,
        BlazeFace works better on three 1080x1080 windows. These overlap by 420
        pixels. (Two windows also work but it's best to have a clean center crop
        in there as well.)

        I also tried 6 windows of size 720x720 (horizontally: 720|360, 360|720;
        vertically: 720|1200, 480|720|480, 1200|720) but that gives many false
        positives when a window has no face in it.

        For a video in portrait orientation (1080x1920), we only take a single
        crop of the top-most 1080 pixels. If we split up the video vertically,
        then we might get false positives again.

        (NOTE: Not all videos are necessarily 1080p but the code can handle this.)

        Arguments:
            frames: NumPy array of shape (num_frames, height, width, 3)
            target_size: (width, height)

        Returns:
            - a new (num_frames * N, target_size[1], target_size[0], 3) array
              where N is the number of tiles used.
            - a list [scale_w, scale_h, offset_x, offset_y] that describes how
              to map the resized and cropped tiles back to the original image 
              coordinates. This is needed for scaling up the face detections 
              from the smaller image to the original image, so we can take the 
              face crops in the original coordinate space.    
        """
        num_frames, H, W, _ = frames.shape

        # Settings for 6 overlapping windows:
        # split_size = 720
        # x_step = 480
        # y_step = 360
        # num_v = 2
        # num_h = 3

        # Settings for 2 overlapping windows:
        # split_size = min(H, W)
        # x_step = W - split_size
        # y_step = H - split_size
        # num_v = 1
        # num_h = 2 if W > H else 1

        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1

        splits = np.zeros((num_frames * num_v * num_h, target_size[1], target_size[0], 3), dtype=np.uint8)

        i = 0
        for f in range(num_frames):
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    crop = frames[f, y:y+split_size, x:x+split_size, :]
                    splits[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        resize_info = [split_size / target_size[0], split_size / target_size[1], 0, 0]
        return splits, resize_info

    def _resize_detections(self, detections, target_size, resize_info):
        """Converts a list of face detections back to the original 
        coordinate system.

        Arguments:
            detections: a list containing PyTorch tensors of shape (num_faces, 17) 
            target_size: (width, height)
            resize_info: [scale_w, scale_h, offset_x, offset_y]
        """
        projected = []
        target_w, target_h = target_size
        scale_w, scale_h, offset_x, offset_y = resize_info

        for i in range(len(detections)):
            detection = detections[i].clone()

            # ymin, xmin, ymax, xmax
            for k in range(2):
                detection[:, k*2    ] = (detection[:, k*2    ] * target_h - offset_y) * scale_h
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_w - offset_x) * scale_w

            # keypoints are x,y
            for k in range(2, 8):
                detection[:, k*2    ] = (detection[:, k*2    ] * target_w - offset_x) * scale_w
                detection[:, k*2 + 1] = (detection[:, k*2 + 1] * target_h - offset_y) * scale_h

            projected.append(detection)

        return projected    
    
    def _untile_detections(self, num_frames, frame_size, detections):
        """With N tiles per frame, there also are N times as many detections.
        This function groups together the detections for a given frame; it is
        the complement to tile_frames().
        """
        combined_detections = []

        W, H = frame_size
        split_size = min(H, W)
        x_step = (W - split_size) // 2
        y_step = (H - split_size) // 2
        num_v = 1
        num_h = 3 if W > H else 1

        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for v in range(num_v):
                x = 0
                for h in range(num_h):
                    # Adjust the coordinates based on the split positions.
                    detection = detections[i].clone()
                    if detection.shape[0] > 0:
                        for k in range(2):
                            detection[:, k*2    ] += y
                            detection[:, k*2 + 1] += x
                        for k in range(2, 8):
                            detection[:, k*2    ] += x
                            detection[:, k*2 + 1] += y

                    detections_for_frame.append(detection)
                    x += x_step
                    i += 1
                y += y_step

            combined_detections.append(torch.cat(detections_for_frame))

        return combined_detections

    def _add_margin_to_detections(self, detections, frame_size, margin=0.2):
        """Expands the face bounding box.

        The face detections often do not include the forehead so we add margin to
        ymin to make the crop area more similar to MTCNN detection.

        Arguments:
            detections: a PyTorch tensor of shape (num_detections, 17)
            frame_size: maximum (width, height)
            margin: a percentage of the bounding box's height

        Returns a PyTorch tensor of shape (num_detections, 17).
        """
        offset = torch.round(margin * (detections[:, 2] - detections[:, 0]))
        detections = detections.clone()
        detections[:, 0] = torch.clamp(detections[:, 0] - offset, min=0)            # ymin
        detections[:, 1] = torch.clamp(detections[:, 1], min=0)                     # xmin
        detections[:, 2] = torch.clamp(detections[:, 2], max=frame_size[1]-1)         # ymax
        detections[:, 3] = torch.clamp(detections[:, 3], max=frame_size[0]-1)         # xmax
        return detections

    def _get_face_detections(self, detections):
        face_detections = []
        for i in range(len(detections)):
            ymin, xmin, ymax, xmax = detections[i, :4].cpu().numpy().astype(np.int)
            facedet = FaceDetection(xmin,ymin,xmax,ymax,
                                    im_width=self.frame_width, im_height=self.frame_height,
                                    score=detections[i, 16].cpu().numpy().astype(np.float),
                                    blazeface_landmarks=detections[i, 4:16].cpu().numpy().astype(np.float32)
                                    )
            if not facedet.isValid(min_face_size=self.min_face_size if self.use_search_limits else None,
                                   search_only_top_part=self.use_search_limits, verbose=self.verbose):
                continue

            face_detections.append(facedet)
        return face_detections