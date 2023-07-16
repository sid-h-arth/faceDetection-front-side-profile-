#library named as computer vision used for face detection programs
import  cv2

#used to classify various faces using the xml file based on the frontal face fetures
#CascadeClassifier is the name of the classifier
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

trained_profile_data=cv2.CascadeClassifier('haarcascade_profileface.xml')
#capture the frames through webcam, 0 is used for default value for webcam webcam
webcam=cv2.VideoCapture(0)

#capture frames through a specified vid
#webcam=cv2.VideoCapture('my_vid.mp4')

#itterate over the webcam video
while True:
    #reading the frames from the webcam
    successfull_frame_read,frame=webcam.read()

    #converting to greyscale
    grey_scaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #getting the coordinates for each frame
    face_coordinates=trained_face_data.detectMultiScale(grey_scaled_img)

    profile_coordinates=trained_profile_data.detectMultiScale(grey_scaled_img)
    #assigning the coordinates to the frame to form a rectangle
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

    for(x,y,w,h) in profile_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

    #displaying the greyscale vid through webcam
    cv2.imshow('vid',frame)
    #automatically preses a key after 1millisec for next frame
    key=cv2.waitKey(1)
    
    #if q or Q is pressed exit the window by breaking the loop
    if key==81 or key==113:
        break

print("completed")