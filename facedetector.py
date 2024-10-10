import cv2
import os

file_path = input("Please enter the file path for the image: ")

if os.path.exists(file_path):
    print("File found! Loading the image...")
else:
    print("File not found. Please check the path and try again.")
    exit()

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread(file_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(image_gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    
    roi_gray = image_gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

image_resized = cv2.resize(image, (800, 600))

cv2.imshow('Image', image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
