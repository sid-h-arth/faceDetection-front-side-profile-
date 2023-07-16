#library named as computer vision used for face detection programs
import  cv2

#used to classify various faces using the xml file based on the frontal face fetures
#CascadeClassifier is the name of the classifier
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

trained_profile_data=cv2.CascadeClassifier('haarcascade_profileface.xml')
#read the image nd store it in 'img'
img=cv2.imread('download.jpg')

#show the image and wait till key is pressed
#cv2.imshow('Sid',img)
#cv2.waitKey()

#convert the image to greyscale for detection purpose
grey_scaled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Sid',grey_scaled_img)
#cv2.waitKey()

#gives the coodinates of the rectangle of the image an dprint it
face_coordinates=trained_face_data.detectMultiScale(grey_scaled_img)

profile_coordinates=trained_profile_data.detectMultiScale(grey_scaled_img)
print(face_coordinates)
print(profile_coordinates)

#to draw the rectangle on the image dynamically
#(x,y,w,h)=face_coordinates[0]
#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

for(x,y,w,h) in profile_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
cv2.imshow('Sid',img)
cv2.waitKey()

print("completed")