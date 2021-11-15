# Online Police Clearance Registration System
This portal is for submitting registration for Online Police Clearance System in Nepal. This service is powered by Artificial Intelligence for different services like Profanity Detection and Image Validation. 
* Enjoy the service at [Link to OPCRS system](https://opcr.nepalpolice.gov.np/)

# Profanity Detection
Users entering profane text in the web interface of any system is a major issue in internet. Here, we are trying to catch such profane activities in the web system.
There are two modules to verify the user input given is accurate or not. They are 
- Uploaded Image Validation
- Input text validation

## Uploaded Image Validation
This module checks whether the image uplaoded by the user in the system is a valid one or not. Whether the image contains the face of the user or the image is random image.
There are two filters for this. They are

- Correct Image size
- Image contains face or not
### Correct Image size
Check whether the uploaded image size meets the requirement or not.

### Image contains face or not
We used MTCNN face detection pretrained model to check whether the uploaded image contains face or not.
This is facilitated through the detect_face module in face_detection/face_validation module.


## Text Validation
In the web system for Nepal Police Clearance Online Regsitration, there were three fields in the user form where we needed to check whether the user typed input is Profane or not.
- Description Field
- First Name Field
- Middle Name Field
- Surname Field

* The main objective is to catch if user typed textual input is profane or not

------- Approach --------
- Developed character level Bidirectional LSTM model for catching the profanity in Name, Middle name and Surname Field. The data for name or not name classification was developed. This data collection contains generally Nepalese names and Nepalese profane words. 
- Developed another simple LSTM model for classifying given description is profane or not using the GLOVE embedding for embedding matrix. 

