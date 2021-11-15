import os
from word_level import WordProfanity
from profanity import CharProfanity
from name_check import CheckName


#For face validation
from scipy import misc
import tensorflow as tf
from face_detection.face_validation.app import detect_face
import numpy as np
import cv2
# import matplotlib.pyplot as plt

from flask import Flask, request, jsonify

myData = {
    
    "data":
    [
        {
            "headName":"Name",
            "isName":"True",
            "headValue":"Jit Bahadur",
            "status":"True",
            "message":""
        },
        {
            "headName":"FathersName",
            "isName":"True",
            "headValue":"Fuck Bd fuck",
            "status":"True",
            "message":""
        },
        {
            "headName":"status",
            "headValue":"Fuck",
            "status":"True",
            "message":""
        },
        {
            "headName":"Discription",
            "headValue":"Hello There",
            "status":"True",
            "message":""
        }
    ]
}


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Face Validate

#This class is written to initialize the tensorflow session and then load the mtcnn models
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
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, os.getcwd() + '/face_detection/face_validation/app/align')

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
    # print("hi")
    for upload in request.files.getlist("file"):
        img_array = np.array(bytearray(upload.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        print(frame)
        image_size = frame.size
        h, w, c = frame.shape

        if c == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        image_ratio = h/w

        #   run detect_face from the mtcnn library
        bounding_boxes, _ = detect_face.detect_face(
                frame, minsize, pnet,
                rnet, onet, threshold, factor)

        print(bounding_boxes)

        nrof_faces = bounding_boxes.shape[0]
        print("No. of faces: ", nrof_faces)

        if nrof_faces > 0:
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1

                roi = frame[int(y1):int(y2),int(x1):int(x2)]

                face_size = roi.size
                print("Face Size: ", face_size)

                cv2.rectangle(frame,(int(x1),int(y1)),(int(x1+w),
                    int(y1+h)),(255,0,0),2)
                print ('Accuracy score', acc)

                face_ratio = face_size/image_size
                print("Face ratio: ", face_ratio)


                ''' Check if face or not'''
                if acc >= 0.98:
                    isFace = True
                else:
                    isFace = False
                
                ''' Check if face ratio to the standard or not'''
                if face_ratio >= 0.15 and face_ratio <= 0.40:
                    isFaceStandard = True
                else:
                    isFaceStandard = False 

                ''' Head to photo ratio check for standardization'''
                if isFace and isFaceStandard:
                    print("Image Accepted")
                    response = {"status":"1", "message":"success"}
                elif not isFace:
                    print("Image not accepted")
                    response = {"status":"0", "message":"Face Not Detected"}
                else:
                    print("Image Not Accepted")
                    response = {"status":"0", "message":"Head should be 40-60% of uploaded image"}
        
        else:
            print("no face detected")
            response = {"status":"0", "message":"No Face Detected"}
    
    return(jsonify(response))








profanity = WordProfanity()
char_profanity = CharProfanity()
pro_response = profanity.predict(myData)
char_response = char_profanity.predict(myData)
# print(pro_response)
name_model = CheckName()
my_response = name_model.check_name(myData)
# print(response)



# # def check_name(data):
# #     for d in data['data']:
# #         name_model = CheckName()
# #         if 'isName' in d:
# #             name = name_model.testing_name_notname(d['headValue'])
# #             if name=="False":
# #                 d['isName'] = "False"
# #                 d['name_msg'] = "Please enter the correct name"
# #     return data


@app.route('/check_profanity', methods = ['POST'])
def postJsonHandler():
    data = request.json

    response = name_model.check_name(data)
    # final_response = 
    # print("RRRRRRRRRRR:",response)

    # for r in response['data']:
    #     name = r['isName']
    #     if name=="False":
    #         return jsonify(response)
    #     else:
    response = char_profanity.predict(data)
    final_response = profanity.predict(response)
    # pro_response = profanity.predict(response)
    return jsonify(final_response)
   
    
    # return jsonify(response)

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host='0.0.0.0', port=8041, debug=True)


