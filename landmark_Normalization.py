from models import mtcnn
import numpy as np
from PIL import Image,ImageDraw
import torch

mtcnn_net = mtcnn.MTCNN()
img = Image.open("2.jpg") #MCTNN detection
timesw = img.width / 256.0 #Normalization
timesh = img.height / 256.0

# img.show()
boxes, probs, points = mtcnn_net.detect(img, landmarks=True)
points = np.reshape(points[0], (5,2))
left_eye = points[:1].reshape(2,1)
right_eye = points[1:2].reshape(2,1)
nose = points[2:3].reshape(2,1)
mouth = points[3:5]

left_eye[0] = int(left_eye[0]/timesw)
left_eye[1] = int(left_eye[1]/timesh)
right_eye[0] = int(right_eye[0]/timesw)
right_eye[1] = int(right_eye[1]/timesh)
nose[0] =  int(nose[0]/timesw)
nose[1] = int(nose[1]/timesh)
mouth[0][0] = int(mouth[0][0]/timesw)
mouth[1][0] = int(mouth[1][0]/timesw)
mouth[0][1] = int(mouth[0][1]/timesh)
mouth[1][1] = int(mouth[1][1]/timesh)

# left_eye[0] = int(left_eye[0])
# left_eye[1] = int(left_eye[1])
# right_eye[0] = int(right_eye[0])
# right_eye[1] = int(right_eye[1])
# nose[0] =  int(nose[0])
# nose[1] = int(nose[1])
# mouth[0][0] = int(mouth[0][0])
# mouth[1][0] = int(mouth[1][0])
# mouth[0][1] = int(mouth[0][1])
# mouth[1][1] = int(mouth[1][1])

img = img.resize((256, 256))
img.save('./mydata/Aligned.jpg')
target = ImageDraw.Draw(img)


#eyes
eye_width = 10
eye_height = 10
left_eye_tuple = (left_eye[0]-eye_width, left_eye[1]-eye_height, left_eye[0]+eye_width, left_eye[1]+eye_height)
target.rectangle(left_eye_tuple, fill='black')
right_eye_tuple = (right_eye[0]-eye_width, right_eye[1]-eye_height, right_eye[0]+eye_width, right_eye[1]+eye_height)
target.rectangle(right_eye_tuple, fill='black')

#nose
nose_size = 8
nose_tuple = (nose[0]-nose_size, nose[1]-nose_size, nose[0]+nose_size, nose[1]+nose_size)
target.rectangle(nose_tuple, fill='black')

#mouth
mouth_height = 7
mouth_tuple = (mouth[0][0], mouth[0][1]-mouth_height, mouth[1][0], mouth[1][1]+mouth_height)
target.rectangle(mouth_tuple, fill='black')


img.show()
img.save('./mydata/landmark_normalization.jpg')

# Save the coordinate
left_eye_coord = np.array([[left_eye[0]-eye_width, left_eye[1]-eye_height, left_eye[0]+eye_width, left_eye[1]+eye_height]]).reshape(1, -1)
np.savetxt("./mydata/left_eye_coord.txt", left_eye_coord, fmt = '%f', delimiter = ',')
right_eye_coord = np.array([[right_eye[0]-eye_width, right_eye[1]-eye_height, right_eye[0]+eye_width, right_eye[1]+eye_height]]).reshape(1, -1)
np.savetxt("./mydata/right_eye_coord.txt", right_eye_coord, fmt = '%f', delimiter = ',')
nose_coord = np.array([[nose[0]-nose_size, nose[1]-nose_size, nose[0]+nose_size, nose[1]+nose_size]]).reshape(1, -1)
np.savetxt("./mydata/nose_coord.txt", nose_coord, fmt = '%f', delimiter = ',')
mouth_coord = np.array([[mouth[0][0], mouth[0][1]-mouth_height, mouth[1][0], mouth[1][1]+mouth_height]]).reshape(1, -1)
np.savetxt("./mydata/mouth_coord.txt", mouth_coord, fmt = '%f', delimiter = ',')