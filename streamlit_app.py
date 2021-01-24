import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import webbrowser
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(hash_funcs={cv2.dnn_Net: hash})
# load face detector
def load_face_detector_model():
    prototxt_path = os.path.sep.join(
        ["face_detector", "deploy.prototxt"])
    weight_path = os.path.sep.join(
        ['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])
    net = cv2.dnn.readNet(prototxt_path, weight_path)
    return net


@st.cache(allow_output_mutation=True)
# load face mask detector model
def load_mask_model():
    mask_model = load_model("mask_detector.model")
    return mask_model

net = load_face_detector_model()
model = load_mask_model()

    
def detect_mask(image):
    label='_'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB
    orig = image.copy() # get a copy of the image
    (h, w) = image.shape[:2] # get image height and weight
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # construct a blob from the image
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)  # pass the blob through the detection, get region that differ in propertes, and the face region
    detection = net.forward() 

    for i in range(0, detection.shape[2]): # loop through the detection

        confidence = detection[0, 0, i, 2] # extract confidence vaalue

        if confidence > 0.50: # if the confidence is greater than the selected confidence from the side bar

            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h]) # get x and y coordinate for the bounding box
            (startX, startY, endX, endY) = box.astype("int") 

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY)) # ensure bounding box does not exceed image frame

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # extract face ROI, convert from BGR to RGB
            face = cv2.resize(face, (224,224))           # resize to 224, 224
            face = img_to_array(face)                    # convert resized face to an array
            face = preprocess_input(face)                # preprocess the array
            face = np.expand_dims(face, axis=0)          # expand array to 2D

            (mask, withoutMask) = model.predict(face)[0] # pass the face through the mask model, detect if there is mask or not

            label = "Mask on" if mask > withoutMask else "No Mask" # define label
            
            color = (0, 255, 0) if label == "Mask on" else (255, 0, 0) # bbox is green if 'mask' else red

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # add label probability 

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.20, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) #display label and bbox rectangle in output frame

        return image # return image
    
    
def detect_mask_image(image):  
    label='_'
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)  #read the image from temporary memory
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image from BGR to RGB
    orig = image.copy() # get a copy of the image
    (h, w) = image.shape[:2] # get image height and weight
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), # construct a blob from the image
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)  # pass the blob through the detection, get region that differ in propertes, and the face region
    detection = net.forward() 

    for i in range(0, detection.shape[2]): # loop through the detection

        confidence = detection[0, 0, i, 2] # extract confidence vaalue

        if confidence > 0.50: # if the confidence is greater than the selected confidence from the side bar

            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h]) # get x and y coordinate for the bounding box
            (startX, startY, endX, endY) = box.astype("int") 

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY)) # ensure bounding box does not exceed image frame

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # extract face ROI, convert from BGR to RGB
            face = cv2.resize(face, (224,224))           # resize to 224, 224
            face = img_to_array(face)                    # convert resized face to an array
            face = preprocess_input(face)                # preprocess the array
            face = np.expand_dims(face, axis=0)          # expand array to 2D

            (mask, withoutMask) = model.predict(face)[0] # pass the face through the mask model, detect if there is mask or not

            label = "Mask on" if mask > withoutMask else "No Mask" # define label
            
            color = (0, 255, 0) if label == "Mask on" else (255, 0, 0) # bbox is green if 'mask' else red

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) # add label probability 

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.20, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2) #display label and bbox rectangle in output frame

        return image, label # return image and label

    
def run_image_detector():       
    image_file = st.file_uploader("Upload image", type=['jpeg', 'jpg', 'png']) # streamlit function to upload file
            
    if image_file is not None: 
            st.sidebar.image(image_file, width=240) # display uploaded image in sidebar
            if st.button("Process"): # button to run algorithm on input image
                image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), 1)
                image, label = detect_mask_image(image) # call mask detection model (image)
                st.image(image, width=420) # display the uploaded image
                st.success('### ' +  label) # display label
                

def run_video_detector():
    run = st.checkbox('Enable webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()

        img = detect_mask(frame) # Call mask detection model (video)

        FRAME_WINDOW.image(img)

    else:
        st.write('Stopped')
           
                
# Helper function for the main page
def main():
    st.sidebar.image('logo.png', width=300)
    
    st.sidebar.markdown("<p style='text-align: left; color: grey;'>As a result of the COVID-19 global pandemic, many governments have made mask wearing mandatory to reduce the transmission of the virus. This web app aims to identify people who are not adhering to the law and gently remind them to please, <font color='skyblue'>wear a mask.</font></p>", unsafe_allow_html=True)
    
    activities = ['Image Detector','Video Detector','The Team']
    choice = st.sidebar.selectbox("Choose the app mode", activities)
    
    url = 'https://github.com/Henry-Lim/maskeraid'

    if st.sidebar.button('Source code'):
        webbrowser.open_new_tab(url)
        
    if choice == 'Image Detector': 
        st.title('maskerAID Image Detector')
        run_image_detector()

    elif choice == "Video Detector": 
        st.title('maskerAID Video Detector')
        run_video_detector()
    
    elif choice == "The Team":
        st.title('The Team')
        st.write('Wei Jie Wong')
        st.write('Alexandros Papadopoulos')
        st.write('Maisey Davidson')
        st.write('Ewan Morrin')
        st.write('Henry Lim')


if __name__ == "__main__": 
    main()