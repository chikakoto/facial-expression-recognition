import cv2
import pickle
import numpy as np
import logging as log

# Emotion labels
emotions = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# Cascade
cascade_file = "haarcascade_frontalface_default.xml"
face_detection = cv2.CascadeClassifier(cascade_file)
log.basicConfig(filename='fer.log',level=log.INFO)

# Load model
## CNN
# model = load_model("../../model/mode.h5", compile=False)
## SVM
model = pickle.load(open("../../models/svm.pkl", "rb"))


font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('Face Expression Detection')
webcam = cv2.VideoCapture(0)
detected = 0

while True:
    
    ret, frame = webcam.read()
    if not ret:
        log.info("no frame captured")
        exit()
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        roi = gray[x:x+w, y:y+h]
        if (roi.shape[0] > 0) & (roi.shape[1] > 0):
            roi = cv2.resize(roi, (48, 48))
            roi = roi.flatten()
            roi = roi.astype("float") / 255.0
#         roi = img_to_array(roi)
#             log.info(roi)
            roi = np.expand_dims(roi, axis=0)
            pred = model.predict(roi)
            pred_label = emotions[pred[0]]
            cv2.putText(frame, pred_label, (x, y), font, 1, (255, 255, 0), 2)
        

    if detected != len(faces):
        detected = len(faces)
#         log.info("faces: "+str(len(faces)))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
webcam.release()
cv2.destroyAllWindows()
