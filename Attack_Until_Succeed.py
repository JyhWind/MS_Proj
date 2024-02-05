from models import mtcnn, inception_resnet_v1
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
# from torchvision import transforms
from PIL import Image,ImageDraw
import os
import cv2



def revise_points(boxes, points):
    points = np.reshape(points[0], (5,2))
    boxes = boxes[0]
    for i in range(0, np.shape(points)[0]):
        points[i][0] = points[i][0] - boxes[0]
        points[i][0] = points[i][0] / (boxes[2] - boxes[0]) * 160
        points[i][1] = points[i][1] - boxes[1]
        points[i][1] = points[i][1] / (boxes[3] - boxes[1]) * 160
    return points.astype(int)

def draw_new_picture(points):
    img_cropped = Image.open("result.jpg")
    target = ImageDraw.Draw(img_cropped)
    left_eye = np.squeeze(points[:1])
    right_eye = np.squeeze(points[1:2])
    nose = np.squeeze(points[2:3])
    mouth = np.squeeze(points[3:5])

    #eyes
    #changed pixels: 20 * 20 * 2 = 800
    eye_width = 20
    eye_height = 20
    l_eye_x0 = left_eye[0]-eye_width
    l_eye_y0 = left_eye[1]-eye_height
    l_eye_x1 = left_eye[0]+eye_width
    l_eye_y1 = left_eye[1]+eye_height
    left_eye_tuple = (l_eye_x0, l_eye_y0, l_eye_x1, l_eye_y1)
    target.rectangle(left_eye_tuple, fill='black')
    r_eye_x0 = right_eye[0]-eye_width
    r_eye_y0 = right_eye[1]-eye_height
    r_eye_x1 = right_eye[0]+eye_width
    r_eye_y1 = right_eye[1]+eye_height
    right_eye_tuple = (r_eye_x0, r_eye_y0, r_eye_x1, r_eye_y1)
    target.rectangle(right_eye_tuple, fill='black')

    #nose 
    #changed pixels: 16 * 16 = 256
    nose_height = 8
    nose_width = 8
    nose_x0 = nose[0]-nose_width
    nose_y0 = nose[0]-nose_height
    nose_x1 = nose[0]+nose_width
    nose_y1 = nose[0]+nose_height
    nose_tuple = (nose_x0, nose_y0, nose_x1, nose_y1)
    target.rectangle(nose_tuple, fill='black')

    #mouth
    #changed pixels: 14 * width
    mouth_height = 7
    mouth_x0 = mouth[0][0]
    mouth_y0 = mouth[0][1]-mouth_height
    mouth_x1 = mouth[1][0]
    mouth_y1 = mouth[1][1]+mouth_height
    mouth_tuple = (mouth_x0, mouth_y0, mouth_x1, mouth_y1)
    target.rectangle(mouth_tuple, fill='black')
    img_cropped.save('./revised_result.jpg')

    changed_pixels = 2 * 2*eye_width * 2*eye_height + 2*nose_width * 2*nose_height + 2*mouth_height * (mouth[1][0] - mouth[0][0])

    #创建遮罩
    mask = np.zeros([160, 160])
    mask[l_eye_x0:l_eye_x1, l_eye_y0:l_eye_y1] = 1
    mask[r_eye_x0:r_eye_x1, r_eye_y0:r_eye_y1] = 1
    mask[nose_x0:nose_x1, nose_y0:nose_y1] = 1
    mask[mouth_x0:mouth_x1, mouth_y0:mouth_y1] = 1
    mask = cv2.GaussianBlur(mask, (3, 3), 0) #高斯平滑
    mask = [[np.copy(mask), np.copy(mask), np.copy(mask)]] #构建符合grad形状的mask
    # print(np.shape(mask))

    return left_eye_tuple, right_eye_tuple, nose_tuple, mouth_tuple, changed_pixels, torch.tensor(mask)


def rgb_to_img(img):
    r = img[0] #160x160
    g = img[1] #160x160
    b = img[2] #160x160

    my_pic = []
    for row_r, row_g, row_b in zip(r, g, b):
        pic_row = []
        for i, j, k in zip(row_r, row_g, row_b):
            pic_row.append([int(i), int(j), int(k)])

        my_pic.append(pic_row)
    my_pic = np.array(my_pic)

    result = Image.fromarray(np.uint8(my_pic))
    return result

def judge(coordx, coordy, targetcoord):
    if coordx > targetcoord[0] and coordx < targetcoord[2] and coordy > targetcoord[1] and coordy < targetcoord[3]:
        return True
    return False

def pixel_is_feature(coordx, coordy, left_eye_coord, right_eye_coord, nose_coord, mouth_coord):
    if judge(coordx, coordy, left_eye_coord) or judge(coordx, coordy, right_eye_coord) or judge(coordx, coordy, nose_coord) or judge(coordx, coordy, mouth_coord):
        return True
    return False
        

# If required, create a face detection pipeline using MTCNN:
mtcnn_net = mtcnn.MTCNN()

# Create an inception resnet (in eval mode):
resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()
resnet.classify = True

def FGSM(img_path):
    img = Image.open(img_path)

    # Get cropped and prewhitened image tensor
    boxes, probs, points = mtcnn_net.detect(img, landmarks=True)
    points = revise_points(boxes, points)
    img_cropped = mtcnn_net(img, save_path="result.jpg")
    left_eye_coord, right_eye_coord, nose_coord, mouth_coord, changed_pixels, mask = draw_new_picture(points)
    rate = changed_pixels / 160 / 160
    print("Features rate = " + str(rate))

    # Calculate embedding (unsqueeze to add batch dimension)
    # img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    # resnet.classify = True
    sq_img = img_cropped.unsqueeze(0)
    my_pic1 = rgb_to_img(np.array(sq_img.squeeze(0) * 255))

    sq_img.requires_grad = True #for gradient calculation
    img_probs = resnet(sq_img)

    # print(img_probs)
    origin_result = img_probs.detach().numpy()[0].tolist()
    dataset_length = str(len(origin_result))
    origin_result = str(origin_result.index(max(origin_result))+1)
    print("Original result: " + str(origin_result.index(max(origin_result))+1) + "/" + dataset_length)

    # ########################################################
    #FGSM ATTACK
    img_probs.backward(torch.ones_like(img_probs))
    fgsm_grad = torch.sign(sq_img.grad.clone()) #归一化
            

    epslion = 0.1
    while True:
        # print("epslion = " + str(epslion))
        mytensor = sq_img.clone() + epslion * torch.mul(fgsm_grad, mask.float())     
        probs = resnet(mytensor)
        fea_result = probs.detach().numpy()[0].tolist()
        fea_result = str(fea_result.index(max(fea_result))+1)
        # print("result considered features: " + str(fea_result.index(max(fea_result))+1) + "/" + str(len(fea_result)))
        if fea_result != origin_result:
            print("Attack with features succeeded. epslion = " + str(epslion))
            break
        epslion += 0.1
  
    old_epslion = epslion
    epslion = 0.01
    while epslion < 1.0:
        mytensor = sq_img.clone()
        mytensor += fgsm_grad * epslion
        probs = resnet(mytensor)
        FGSM_result = probs.detach().numpy()[0].tolist()
        FGSM_result = str(FGSM_result.index(max(FGSM_result))+1)
        # print("result not considered features: " + str(FGSM_result.index(max(FGSM_result))+1) + "/" + str(len(FGSM_result)))
        if FGSM_result != origin_result:
            print("Attack succeeded. epslion = " + str(epslion))
            break
        epslion += 0.01
    result = (epslion - (old_epslion / (1 / rate))) / epslion
    result *= 100
    result = round(result, 2)
    print("Improvement Rate: " + str(result) + "%")
    return result

for i in range(1, 2):
    path = './mydata/' + str(i) + '/'
    r = open(path + 'result.txt', 'w+')
    files = os.listdir(path)
    for file in files:
        print("Now working on " + file)
        # try:
        result = FGSM(path + file)
        r.write("file " + file + " Improvement Rate: " + str(result) + "%\n")
        print("————————————————————————————————————")
        # except:
        #     print('file ' + file + ' error')
        break
    r.close()

        