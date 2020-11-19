import cv2
import os

# step 1 : Loading haarcascade face algorithm
alg = "haarcascade_frontalface_alt.xml"
haar = cv2.CascadeClassifier(alg)

#Step 2 : Initializing Camera
cam = cv2.VideoCapture(0)

dataset = "dataset"
name = "champ"

path = os.path.join(dataset , name)
if not os.path.isdir(path):
    os.mkdir(path)

(width , height) = (130,100)
count = 1

#Step 3 : reading frame from camera
while (count<30):
    print(count)
    _,img = cam.read()

    #step 4: Converting color image to gray scale img
    grayImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY )

    #step 5 : obtaining face coordinates by passing algorithm
    # faces = face_cascade.detectMultiScale( src , scalefactor , minNeighbours 0
    faces = haar.detectMultiScale(grayImg , 1.3 , 4 )

    #step 6 :Drawing rectangle on face coordinates
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        onlyFace = grayImg[ y:y+h , x:x+w ]
        resizeImg = cv2.resize(grayImg , (width , height ))
        #1st %s is for path and 2nd %s is for count no.
        cv2.imwrite( "%s/%s.jpg" %(path,count) ,resizeImg)
        count+=1

    #step 7 : display the output frame
    cv2.imshow("FaceDetection" , img )

    key = cv2.waitKey(10)
    if key == 27:     #27 is for escape
        break

print("face captured successfully")
cam.release()
cv2.destroyAllWindows()
    
