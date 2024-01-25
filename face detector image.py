import cv2
from deepface import DeepFace

imgpath = "XXXXX"

image = cv2.imread(imgpath)

age_pred = DeepFace.analyze(image, actions=["age"],)
age = age_pred[0].get("age")
gender_pred = DeepFace.analyze(image, actions=["gender"])
gender = (gender_pred[0]["dominant_gender"])
emotion_pred = DeepFace.analyze(image, actions=["emotion"])
emotion = (emotion_pred[0]["dominant_emotion"]).capitalize()
race_pred = DeepFace.analyze(image, actions=["race"])
race = (race_pred[0]["dominant_race"]).capitalize()

print(f"Age: {age}, \nGender: {gender}, \nEmotion: {emotion}, \nRace: {race}.")