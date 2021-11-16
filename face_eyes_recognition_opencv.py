# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:02:14 2021
@author: Manuel
"""
import cv2

# step_1 load cascades values
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_eye.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  'haarcascade_smile.xml')


# step_2 Defining a function for detection
def detect(grayscale_image, input_image):
    #image, reduction_factor of the image, minNeighbors zone to be accepted
    faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5 )
    #4 points of the rectangle 
    #(x, y) top left, w,h width and height
    for(x, y, w, h) in faces:
        cv2.rectangle(input_image, (x,y), (x+w, y+h), (255,0,0), 2 )
        cropped_face_gray = grayscale_image[y:y+h, x:x+w]
        cropped_face = input_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(cropped_face_gray, 1.1, 22 )
        for(eye_x, eye_y, eye_w, eye_h) in eyes:
            cv2.rectangle(cropped_face, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (0,255,0), 2)
        smiles = smile_cascade.detectMultiScale(cropped_face_gray, 1.7, 22 )
        for(smile_x, smile_y, smile_w, smile_h) in smiles:
            cv2.rectangle(cropped_face, (smile_x, smile_y), (smile_x+smile_w, smile_y+smile_h), (0,0,255), 2)
    return input_image

#face recognition script using webcam
webcam_capture = cv2.VideoCapture(0)

while True:
    _, image_input = webcam_capture.read()
    gray_image =  cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    output_image = detect(gray_image, image_input)
    cv2.imshow('Video', output_image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_capture.release()
cv2.destroyAllWindows()
