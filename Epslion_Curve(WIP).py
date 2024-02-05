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

def draw_new_picture(points, img, save_path):
    # img_cropped = Image.open("result.jpg")

    img_cropped = rgb_to_img(img)
    target = ImageDraw.Draw(img_cropped)
    left_eye = np.squeeze(points[:1])
    right_eye = np.squeeze(points[1:2])
    nose = np.squeeze(points[2:3])
    mouth = np.squeeze(points[3:5])

    #eyes
    #changed pixels: (20*2) * (20*2) * 2 = 1600
    eye_width = 20
    eye_height = 20
    l_eye_x0 = max(left_eye[0]-eye_width, 0)
    l_eye_y0 = max(left_eye[1]-eye_height, 0)
    l_eye_x1 = min(left_eye[0]+eye_width, 159)
    l_eye_y1 = min(left_eye[1]+eye_height, 159)
    left_eye_tuple = (l_eye_x0, l_eye_y0, l_eye_x1, l_eye_y1)
    target.rectangle(left_eye_tuple, fill='black')
    r_eye_x0 = max(right_eye[0]-eye_width, 0)
    r_eye_y0 = max(right_eye[1]-eye_height, 0)
    r_eye_x1 = min(right_eye[0]+eye_width, 159)
    r_eye_y1 = min(right_eye[1]+eye_height, 159)
    right_eye_tuple = (r_eye_x0, r_eye_y0, r_eye_x1, r_eye_y1)
    target.rectangle(right_eye_tuple, fill='black')

    #nose 
    #changed pixels: (2*8) * (2*8) = 256
    nose_height = 8
    nose_width = 8
    nose_x0 = max(nose[0]-nose_width, 0)
    nose_y0 = max(nose[0]-nose_height, 0)
    nose_x1 = min(nose[0]+nose_width, 159)
    nose_y1 = min(nose[0]+nose_height, 159)
    nose_tuple = (nose_x0, nose_y0, nose_x1, nose_y1)
    target.rectangle(nose_tuple, fill='black')

    #mouth
    #changed pixels: (2*7) * width
    mouth_height = 7
    mouth_x0 = mouth[0][0]
    mouth_y0 = max(mouth[0][1]-mouth_height, 0)
    mouth_x1 = mouth[1][0]
    mouth_y1 = min(mouth[1][1]+mouth_height, 159)
    mouth_tuple = (mouth_x0, mouth_y0, mouth_x1, mouth_y1)
    target.rectangle(mouth_tuple, fill='black')
    # img_cropped.save("./Consider_Trip/new/0.025/1/w.jpg") #遮罩图片保存
    # img_cropped.show()

    changed_pixels = 2 * 2*eye_width * 2*eye_height + 2*nose_width * 2*nose_height + 2*mouth_height * (mouth[1][0] - mouth[0][0])
    coverage = changed_pixels / 160.0 / 160.0

    #创建遮罩
    mask = np.zeros([160, 160])
    mask[l_eye_y0:l_eye_y1, l_eye_x0:l_eye_x1] = 1
    mask[r_eye_y0:r_eye_y1, r_eye_x0:r_eye_x1] = 1
    mask[nose_y0:nose_y1, nose_x0:nose_x1] = 1
    mask[mouth_y0:mouth_y1, mouth_x0:mouth_x1] = 1
    mask = cv2.GaussianBlur(mask, (3, 3), 0) #Gaussian
    mask = [[np.copy(mask), np.copy(mask), np.copy(mask)]] #构建符合grad形状的mask
    # print(np.shape(mask))
    return coverage, torch.tensor(np.array(mask)) #faster


def rgb_to_img(img):
    img = np.array(img*128 + 127.5)
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

def cal_psnr(im1, im2):
      mse = (np.abs(im1 - im2) ** 2).mean()
      psnr = 10 * np.log10(255 * 255 / mse)
      return psnr

def FGSM(img_path, e, i, Original_Method, save_path):
    attack_succeeded = 0
    img = Image.open(img_path)

    # Get cropped and prewhitened image tensor
    boxes, probs, points = mtcnn_net.detect(img, landmarks=True)
    img_cropped = mtcnn_net(img)


    # Calculate embedding (unsqueeze to add batch dimension)
    # img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    
    sq_img = img_cropped.unsqueeze(0)
    sq_img.requires_grad = True #for gradient calculation
    img_probs = resnet(sq_img)

    # print(img_probs)
    # origin_result = img_probs.detach().numpy()[0].tolist()
    # dataset_length = str(len(origin_result))
    # origin_result = str(origin_result.index(max(origin_result))+1)
    # print("Original result: " + str(origin_result.index(max(origin_result))+1) + "/" + dataset_length)

    # ########################################################
    #FGSM ATTACK
    img_probs.backward(torch.ones_like(img_probs))
    fgsm_grad = torch.sign(sq_img.grad.clone()) #归一化
    epslion = e
    
    if Original_Method == False: #新方法
        points = revise_points(boxes, points)
        coverage, mask = draw_new_picture(points, img_cropped, save_path)
        mytensor = sq_img.clone() + epslion * torch.mul(fgsm_grad, mask.float())
    else:
        mytensor = sq_img.clone() + epslion * fgsm_grad #原方法
        coverage = 1
        
    rgb_to_img(mytensor.squeeze(0).detach()).save(save_path)
        
    probs = resnet(mytensor)
    fea_result = probs.detach().numpy()[0].tolist()
    fea_result = str(fea_result.index(max(fea_result))+1)
    # print("result considered features: " + str(fea_result.index(max(fea_result))+1) + "/" + str(len(fea_result)))
    if fea_result != real_result[i]:
        attack_succeeded = 1

    
    pic1 = np.array(img_cropped.squeeze(0) * 128 + 127.5)
    pic2 = mytensor.squeeze(0) * 128 + 127.5
    pic2 = np.array(pic2.detach())
    PSNR = cal_psnr(pic1, pic2)
    return coverage, attack_succeeded, PSNR

#测试原本标签
real_result = []
for i in range(1, 31):
    path = './mydata/' + str(i) + '/'
    files = os.listdir(path)
    img = Image.open(path + files[0])
    img_probs = resnet(mtcnn_net(img).unsqueeze(0))
    origin_result = img_probs.detach().numpy()[0].tolist()
    real_result.append(str(origin_result.index(max(origin_result))+1))



is_FGSM = True #是否是fgsm
data_size = 100

for e in np.arange(0.03, 0.07, 0.001):
    if is_FGSM:
        result_path = './Consider_Trip/fgsm/'+str(round(e, 3))
    else:
        result_path = './Consider_Trip/new/'+str(round(e, 3))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    r = open(result_path+'/result.txt', 'w') #追加改成a
    average_coverage  = 0
    PSNRs = 0
    num = 0
    num_succeeded = 0
    print("epslion = " + str(round(e, 3)))
    for i in range(1, 31):
        if(num >= data_size):
            break
        path = './mydata/' + str(i) + '/'
        files = os.listdir(path)
        # print("Now working on " + str(i))
        for file in files:
            try:
                if is_FGSM:
                    f_path =  './Consider_Trip/fgsm/'+str(round(e, 3))+'/'+str(i)
                else:
                    f_path =  './Consider_Trip/new/'+str(round(e, 3))+'/'+str(i)
                if not os.path.exists(f_path):
                    os.makedirs(f_path)
                coverage, attack_succeeded, PSNR = FGSM(path + file, e, i-1, is_FGSM, f_path+'/'+file)
                PSNRs += PSNR
                num += 1
                num_succeeded += attack_succeeded
                average_coverage  += coverage 
                if(num >= data_size):
                    break
            except:
                print(str(i) + '-file ' + file + ' error')

    r.write('epslion =     ' + str(round(e, 3)) + ', ')
    r.write("succeeded rate: " + str(num_succeeded) + "/" + str(num)) 
    average_coverage /= data_size
    average_coverage *= 100
    PSNRs /= data_size
    # r.write("average_coverage = " + str(round(average_coverage , 2)) + "%" + '\n')
    r.write(", average_PSNR = " + str(round(PSNRs, 2)) + '\n')
    r.close()

