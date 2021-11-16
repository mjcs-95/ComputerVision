# ComputerVision
 Some computer vision code

To make the face_eye detector to work you need to download the 2 cascades files(links attached in the code) and put the files in the same folder that the script.

In the file facedetector.py:
 
 If you get the error 'SystemError: <class 'cv2.CascadeClassifier'> returned a result with an error set' then use the command 'pip install opencv-contrib-python' and change the lines were we load the cascadefiles to :

face_cascade = cv2.CascadeClassifier( cv2.data.haarcascades +  'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier( cv2.data.haarcascades +  'haarcascade_eye.xml')
 
