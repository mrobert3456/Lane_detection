import pandas as pd
import numpy as np
import h5py
import cv2
from tensorflow.keras.models import load_model

class TrafficSignDetector:
    def __init__(self):
        self.model = load_model('ts/model_1_ts_gray.h5')

        # loading trained weights
        self.model.load_weights('ts/w_1_dataset_ts_gray_norm.h5')

        # loading class names
        self.labels = pd.read_csv('osztalyok.csv', sep=',',encoding='latin-1')

        # Converting into Numpy array
        self.labels = np.array(self.labels.loc[:, 'SignName']).flatten()

        # open dataset -> it is only needed for the mean subraction dataset, or for std dataset
        #with h5py.File('ts/mean_rgb_dataset_ts.hdf5', 'r') as f:
        #    self.mean_rgb = f['mean']  # HDF5 dataset
        #    self.mean_rgb = np.array(self.mean_rgb)  # Numpy arrays


        # ----------------------darknet------------------------

        self.path_to_weights = 'ts/yolov3_ts_traine_8000.weights'
        self.path_to_cfg = 'ts/yolov3_ts_test.cfg'

        # Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
        self.network = cv2.dnn.readNetFromDarknet(self.path_to_cfg, self.path_to_weights)

        # To use with GPU
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        # Minimum probability to eliminate weak detections
        self.probability_minimum = 0.6
        # Setting threshold to filtering weak bounding boxes by non-maximum suppression
        self.threshold = 0.2

        # Getting names of all YOLO v3 layers
        self.layers_all = self.network.getLayerNames()

        # Getting only detection YOLO v3 layers
        self.layers_names_output = [self.layers_all[i - 1] for i in self.network.getUnconnectedOutLayers()]

        #writer = None
        self.h, self.w = None, None

    def ROI(self,img):
        """Gets the ROI from the image"""
        mask = np.zeros_like(img)

        imshape = img.shape

        vertices = np.array(
            [[(0, imshape[0]*0.5), (imshape[1]*.1 , imshape[0]*.1 ), (imshape[1]*.8 , imshape[0]*.1),
              (imshape[1], imshape[0]*0.5)]], dtype=np.int32)  # creates an array with the trapezoids verticies

        cv2.fillPoly(mask, vertices, (255,) * 3)
        masked_image = cv2.bitwise_and(img, mask)  # crops the original image with the mask


        return masked_image

    def DetectSign(self,img):
        """
        Detection traffic signs in the input image
        """
        frame = self.ROI(img)
        if self.w is None or self.h is None:
            # Slicing two elements from tuple
            h, w = frame.shape[:2]

        # Blob from current frame -> this preprocessing the image to be normalized, resized and converts into RGB
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass with blob through output layers
        self.network.setInput(blob)
        output_from_network = self.network.forward(self.layers_names_output)

        # Lists for detected bounding boxes, confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # class probabilities
                scores = detected_objects[5:]
                # index of the most probable class
                class_current = np.argmax(scores)
                # Getting probability values for current class
                confidence_current = scores[class_current]


                if confidence_current > self.probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates of bounding box
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))


                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        # Implementing non-maximum suppression of given bounding boxes
        # this will get only the relevant bounding boxes (there might be more which crosses each other, and etc)
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)

        # Checking if there is any detected object been left
        if len(results) > 0:

            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Cut fragment with Traffic Sign
                c_ts = frame[y_min:y_min + int(box_height), x_min:x_min + int(box_width), :]
                if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                    pass
                else:
                    # Blob from current frame -> this preprocessing the image to be normalized, resized and converts into RGB
                    blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(48, 48), swapRB=True, crop=False)
                    #blob_ts[0] = blob_ts[0, :, :, :] - self.mean_rgb # only needed for mean subtraction
                    #blob_ts[0] = blob_ts[0, :, :, :] / self.std_rgb # only needed for std
                    blob_ts = blob_ts.transpose(0, 2, 3, 1)

                    # CONVERTING GRAY, CAN BE OMITTED IF YOU ARE USING RGB MODELL
                    blob_ts = np.squeeze(blob_ts)  # shape (48,48,3)

                    blob_ts = cv2.cvtColor(blob_ts, cv2.COLOR_RGB2GRAY)  # shape (48,48)

                    blob_ts = blob_ts[np.newaxis, :, :, np.newaxis]  # shape (1,48,48,1)


                    scores = self.model.predict(blob_ts)

                    # highest probability class
                    prediction = np.argmax(scores)

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(img, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  (0, 255, 0), 2)


                    text_box_current = '{}: {:.4f}'.format(self.labels[prediction],
                                                           confidences[i])


                    cv2.putText(img, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return img