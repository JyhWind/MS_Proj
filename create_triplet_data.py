from models import mtcnn
from PIL import Image
import os

mtcnn_net = mtcnn.MTCNN()
roots = os.listdir("./mydata")

for root in roots:
    path = "./mydata/" + root
    files = os.listdir(path)
    num = 0
    for file in files:
        img = Image.open(path + "/" + file)
        img_cropped = mtcnn_net(img, save_path = "../data/g_data/" + root + "/" + file)
        num += 1
        # if num >= 50:
        #     break   



# for file in files_1:
#     img = Image.open("../data/1/" + file)
#     img_cropped = mtcnn_net(img, save_path = "../data/1/" + file)
    
# for file in files_2:
#     img = Image.open("../data/2/" + file)
#     img_cropped = mtcnn_net(img, save_path = "../data/2/" + file)