import cv2
import os
import numpy

# Loading haarcascade face algorithm
alg = "haarcascade_frontalface_alt.xml"
haar = cv2.CascadeClassifier(alg)

#Initializing Camera
cam = cv2.VideoCapture(0)

dataset = "dataset"
(images,labels,names,id) = ( [] , [] , {}, 0)

for( subdirs , dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dataset , subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label =id
            images.append(cv2.imread(path , 0))
            labels.append(int(label))
            #print(labels)
        id += 1

(width,height) = (130,100)
(images , labels) = [numpy.array(lis) for lis in [images,labels]]
print(images , labels)

#print(dir (cv2.face))
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images , labels )

print("Training Completed")

count = 0

#reading frame from camera
while True:
    (_,img) = cam.read()

    #Converting color image to gray scale img
    grayImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY )

    #obtaining face coordinates by passing algorithm
    # faces = face_cascade.detectMultiScale( src , scalefactor , minNeighbours 0
    faces = haar.detectMultiScale(grayImg , 1.3 , 4 )

    #Drawing rectangle on face coordinates
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2 )
        face = grayImg[ y:y+h , x:x+w ]
        face_resize = cv2.resize(face , (width , height ))

        prediction = model.predict(face_resize)
        cv2.rectangle(img , (x,y) ,(x+w,y+h) , (0,255,0) , 2 )
        if prediction[1]<800:
            cv2.putText(img, '%s - %.0f' %(names[prediction[0]] , prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255))
            count = 0
        else :
            count +=1
            cv2.putText(img,'Unknown' , (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0) , 2 )
            if(count>100):
                        print("Unknown")
                        cv2.imwrite("input.jpg",img)
                        count = 0

    #display the output frame
    cv2.imshow("OpenCV" , img )

    key = cv2.waitKey(10)
    if key == 27:     #27 is for escape
        break

cam.release()
cv2.destroyAllWindows()
    
