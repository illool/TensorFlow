# coding: utf-8
import cv2
import sys
'''
# Get user supplied values
#imagePath = sys.argv[1]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #1
# Read the image
#image = cv2.imread(imagePath)#2
image = cv2.imread("1.jpg")#2
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#3
# Detect faces in the image
#faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5,5),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
) #4
print("Found {0} faces!".format(len(faces)))#5
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) #6
cv2.imshow("Faces found", image)#7
cv2.waitKey(0) #8
'''


cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #1
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)
        #write the flipped frame
        #out.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5,5),
            #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            ) #4
        print("Found {0} faces!".format(len(faces)))#5
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 125, 0), 1) #6
        cv2.imshow("Faces found", frame)#7
        #cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
