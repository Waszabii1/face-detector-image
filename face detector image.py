import cv2
from deepface import DeepFace
import numpy as np

imgpath = "C:\\Users\\mario\\Coding_Projeccts\\face detector image\\pictures\\download.jpg"

image = cv2.imread(imgpath)

age = DeepFace.analyze(image, actions=["age"])
print(age)
gender = DeepFace.analyze(image, actions=["gender"])
print(gender[0]["dominant_gender"])
emotion = DeepFace.analyze(image, actions=["emotion"])
print(emotion[0]["dominant_emotion"])
race = DeepFace.analyze(image, actions=["race"])
print(race[0]["dominant_race"])
