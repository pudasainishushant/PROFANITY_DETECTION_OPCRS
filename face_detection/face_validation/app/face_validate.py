
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import os
import detect_face
import numpy as np

import cv2
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, send_from_directory, jsonify


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


class LoadModel():
    """  Importing and running isolated TF graph """
    def __init__(self):
        # Create local graph and use it in the session
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
            # self.sess = tf.Session(config=tf.ConfigProto())

            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, os.getcwd() + '/align')

    def nets(self):
        return(self.pnet, self.rnet, self.onet)

model = LoadModel()
pnet, rnet, onet = model.nets()

#setup facenet parameters
gpu_memory_fraction = 1.0
minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#Recognize
@app.route("/validate", methods=["GET","POST"])
def validate():

    for upload in request.files.getlist("file"):

        img_array = np.array(bytearray(upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)

        image_size = frame.size
        h, w, _ = frame.shape
        image_ratio = h/w

        print("Image Size: ", image_size)

        #   run detect_face from the facenet library
        bounding_boxes, _ = detect_face.detect_face(
                frame, minsize, pnet,
                rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces > 0:
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1

                roi = frame[int(y1):int(y2),int(x1):int(x2)]

                face_size = roi.size
                print("Face Size: ", face_size)

                # cv2.imshow('ROI', roi)

                #   plot the box using cv2
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x1+w),
                    int(y1+h)),(255,0,0),2)
                print ('Accuracy score', acc)

                face_ratio = face_size/image_size
                print("Face ratio: ", face_ratio)
                if image_ratio > 1.25 and image_ratio < 1.3:
                    if face_ratio >= 0.2 and face_ratio <= 0.3:
                        print("Image Accepted")
                        response = {"face":"Detected", "image_status":"Valid Image Size"}
                    else:
                        print("Image Not Accepted")
                        response = {"face":"Detected", "image_status":"Invalid Image Size"}
                else:
                    print("Image Not Accepted")
                    response = {"face":"Detected", "image_status":"Invalid Image Size"}

            #   save a new file with the boxed face
            cv2.imwrite('detected_face.jpg', frame)
            #   show the boxed face

            # cv2.imshow('Face Detected', frame)
            # cv2.waitKey(0)
        
        else:
            print("no face detected")
            response = {"face":"Not Detected", "image_status":"No Face"}
    
    return(jsonify(response))



# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8012, debug=True)