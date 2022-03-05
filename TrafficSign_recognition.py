import pandas as pd
import numpy as np
import h5py
import cv2
import io
import os
from tensorflow.keras.models import load_model

class TrafficSignDetector:
    def __init__(self):
        self.model = load_model('ts' + '/' + 'model_ts_rgb.h5')

        # loading trained weights
        self.model.load_weights('ts' + '/' + 'w_1_ts_rgb_255_mean.h5')

        # loading class names
        self.labels = pd.read_csv('osztalyok.csv', sep=',')

        # Converting into Numpy array
        self.labels = np.array(self.labels.loc[:, 'SignName']).flatten()

        # open dataset
        with h5py.File('ts' + '/' + 'mean_rgb_dataset_ts.hdf5', 'r') as f:
            # Extracting saved array for Mean Image
            self.mean_rgb = f['mean']  # HDF5 dataset

            # Converting it into Numpy array
            self.mean_rgb = np.array(self.mean_rgb)  # Numpy arrays

        # ----------------------darknet------------------------

        self.path_to_weights = 'ts/yolov3_ts_train_final.weights'
        self.path_to_cfg = 'ts/yolov3_ts_test.cfg'

        # Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
        self.network = cv2.dnn.readNetFromDarknet(self.path_to_cfg, self.path_to_weights)

        # To use with GPU
        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        # Minimum probability to eliminate weak detections
        self.probability_minimum = 0.9
        # Setting threshold to filtering weak bounding boxes by non-maximum suppression
        self.threshold = 0.2

        # Generating colours for bounding boxes
        # randint(low, high=None, size=None, dtype='l')
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        # Check point

        # Getting names of all YOLO v3 layers
        self.layers_all = self.network.getLayerNames()

        # Check point
        # print(layers_all)

        # Getting only detection YOLO v3 layers that are 82, 94 and 106
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
        frame = self.ROI(img)
        #return frame
        if self.w is None or self.h is None:
            # Slicing two elements from tuple
            h, w = frame.shape[:2]

        # Blob from current frame
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Forward pass with blob through output layers
        self.network.setInput(blob)
        # start = time.time()
        output_from_network = self.network.forward(self.layers_names_output)
        # end = time.time()

        # Increasing counters
        #  f += 1
        # t += end - start

        # Spent time for current frame
        # print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        # Lists for detected bounding boxes, confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions by minimum probability
                if confidence_current > self.probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        # Implementing non-maximum suppression of given bounding boxes
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)

        # Checking if there is any detected object been left
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Bounding box coordinates, its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Cut fragment with Traffic Sign
                c_ts = frame[y_min:y_min + int(box_height), x_min:x_min + int(box_width), :]
                if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                    pass
                else:
                    # Getting preprocessed blob with Traffic Sign of needed shape
                    blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(48, 48), swapRB=True, crop=False)
                    # blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                    blob_ts = blob_ts.transpose(0, 2, 3, 1)
                    # plt.imshow(blob_ts[0, :, :, :])
                    # plt.show()

                    # Feeding to the Keras CNN model to get predicted label among 43 classes
                    scores = self.model.predict(blob_ts)

                    # Scores is given for image with 43 numbers of predictions for each class
                    # Getting only one class with maximum value
                    prediction = np.argmax(scores)
                    # print(labels['SignName'][prediction])

                    # Colour for current bounding box
                    colour_box_current = self.colours[class_numbers[i]].tolist()

                    # Drawing bounding box on the original current frame
                    cv2.rectangle(img, (x_min, y_min),
                                  (x_min + box_width, y_min + box_height),
                                  (0, 255, 0), 2)

                    # Preparing text with label and confidence for current bounding box
                    text_box_current = '{}: {:.4f}'.format(self.labels[prediction],
                                                           confidences[i])

                    # Putting text with label and confidence on the original image
                    cv2.putText(img, text_box_current, (x_min, y_min - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
        return img