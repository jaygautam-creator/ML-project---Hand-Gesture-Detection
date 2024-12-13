# # import cv2
# # from cvzone.HandTrackingModule import HandDetector
# # from cvzone.ClassificationModule import Classifier
# # import numpy as np
# # import math

# # cap = cv2.VideoCapture(0)
# # detector = HandDetector(maxHands=1)
# # classifier = Classifier("/Users/jaygautam/Desktop/converted_keras/keras_model.h5" , "/Users/jaygautam/Desktop/converted_keras/labels.txt")
# # offset = 20
# # imgSize = 300
# # counter = 0

# # # labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"] 
# # labels = ["Hello","I love you","No","Thank you","Yes"]


# # while True:
# #     success, img = cap.read()
# #     imgOutput = img.copy()
# #     hands, img = detector.findHands(img)
# #     if hands:
# #         hand = hands[0]
# #         x, y, w, h = hand['bbox']

# #         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

# #         imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
# #         imgCropShape = imgCrop.shape

# #         aspectRatio = h / w

# #         if aspectRatio > 1:
# #             k = imgSize / h
# #             wCal = math.ceil(k * w)
# #             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
# #             imgResizeShape = imgResize.shape
# #             wGap = math.ceil((imgSize-wCal)/2)
# #             imgWhite[:, wGap: wCal + wGap] = imgResize
# #             prediction , index = classifier.getPrediction(imgWhite, draw= False)
# #             print(prediction, index)

# #         else:
# #             k = imgSize / w
# #             hCal = math.ceil(k * h)
# #             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
# #             imgResizeShape = imgResize.shape
# #             hGap = math.ceil((imgSize - hCal) / 2)
# #             imgWhite[hGap: hCal + hGap, :] = imgResize
# #             prediction , index = classifier.getPrediction(imgWhite, draw= False)

       
# #         cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

# #         cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
# #         cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

# #         cv2.imshow('ImageCrop', imgCrop)
# #         cv2.imshow('ImageWhite', imgWhite)

# #     cv2.imshow('Image', imgOutput)
# #     cv2.waitKey(1)
    
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tensorflow as tf

# # Ensure compatibility
# print(tf.__version__)
# print(tf.keras.__version__)

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("/Users/jaygautam/Desktop/converted_keras/keras_model.h5", "/Users/jaygautam/Desktop/converted_keras/labels.txt")

# offset = 20
# imgSize = 300
# counter = 0

# labels = ["Hello", "I love you", "No", "Thank you", "Yes"]

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']

#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape

#         aspectRatio = h / w

#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             print(prediction, index)
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)

#         cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

#         cv2.imshow('ImageCrop', imgCrop)
#         cv2.imshow('ImageWhite', imgWhite)

#     cv2.imshow('Image', imgOutput)
#     cv2.waitKey(1)

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf

# Ensure compatibility
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Custom DepthwiseConv2D layer definition
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Pop the 'groups' argument before calling the super constructor
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Paths
model_path = "/Users/jaygautam/Downloads/converted_keras/keras_model.h5"
labels_path = "/Users/jaygautam/Downloads/converted_keras/labels.txt"

# Load the model with custom DepthwiseConv2D layer
model = tf.keras.models.load_model(model_path, custom_objects={'CustomDepthwiseConv2D': CustomDepthwiseConv2D})
print("Model loaded successfully!")

# Load labels
with open(labels_path, 'r') as f:
    labels = f.read().splitlines()

# Initialize video capture and classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, labels_path)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        if imgCrop.size == 0:
            continue  # Skip if the cropped image is empty
        
        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        except Exception as e:
            print(f"Error processing image: {e}")

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
