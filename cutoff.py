# Based on Normalized input 256 * 256
from PIL import Image
import numpy as np
img = Image.open("3.jpg").resize((256, 256))
left_eye_coord = np.loadtxt('./mydata/left_eye_coord.txt', delimiter = ',')
right_eye_coord = np.loadtxt('./mydata/right_eye_coord.txt', delimiter = ',')
nose_coord = np.loadtxt('./mydata/nose_coord.txt', delimiter = ',')
mouth_coord = np.loadtxt('./mydata/mouth_coord.txt', delimiter = ',')

cropped_left_eye = img.crop((left_eye_coord[0], left_eye_coord[1], left_eye_coord[2], left_eye_coord[3]))
cropped_right_eye = img.crop((right_eye_coord[0], right_eye_coord[1], right_eye_coord[2], right_eye_coord[3]))
cropped_nose = img.crop((nose_coord[0], nose_coord[1], nose_coord[2], nose_coord[3]))
cropped_mouth = img.crop((mouth_coord[0], mouth_coord[1], mouth_coord[2], mouth_coord[3]))

cropped_left_eye.save("./mydata/left_eye.jpg") #20 * 20
cropped_right_eye.save("./mydata/right_eye.jpg") #20 * 20
cropped_nose.save("./mydata/nose.jpg") #16 * 16
cropped_mouth.save("./mydata/mouth.jpg") #14 * 14