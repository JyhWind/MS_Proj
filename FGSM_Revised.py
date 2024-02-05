from models import mtcnn, inception_resnet_v1
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image,ImageDraw

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
    left_eye = points[:1].reshape(2,1)
    right_eye = points[1:2].reshape(2,1)
    nose = points[2:3].reshape(2,1)
    mouth = points[3:5]

    #eyes
    #changed pixels: 20 * 20 * 2 = 800
    eye_width = 10
    eye_height = 10
    left_eye_tuple = (left_eye[0]-eye_width, left_eye[1]-eye_height, left_eye[0]+eye_width, left_eye[1]+eye_height)
    target.rectangle(left_eye_tuple, fill='black')
    right_eye_tuple = (right_eye[0]-eye_width, right_eye[1]-eye_height, right_eye[0]+eye_width, right_eye[1]+eye_height)
    target.rectangle(right_eye_tuple, fill='black')

    #nose 
    #changed pixels: 16 * 16 = 256
    nose_size = 8
    nose_tuple = (nose[0]-nose_size, nose[1]-nose_size, nose[0]+nose_size, nose[1]+nose_size)
    target.rectangle(nose_tuple, fill='black')

    #mouth
    #changed pixels: 14 * width
    mouth_height = 7
    mouth_tuple = (mouth[0][0], mouth[0][1]-mouth_height, mouth[1][0], mouth[1][1]+mouth_height)
    target.rectangle(mouth_tuple, fill='black')
    img_cropped.save('./revised_result.jpg')

    changed_pixels = 2 * 2*eye_width * 2*eye_height + 2*nose_size * 2*nose_size + 2*mouth_height * (mouth[1][0] - mouth[0][0])
    return left_eye_tuple, right_eye_tuple, nose_tuple, mouth_tuple, changed_pixels


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


# If required, create a face detection pipeline using MTCNN:
mtcnn_net = mtcnn.MTCNN()

# Create an inception resnet (in eval mode):
resnet = inception_resnet_v1.InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open("./2.jpg")

# Get cropped and prewhitened image tensor
# convert_tensor = transforms.ToTensor()
# img_cropped = convert_tensor(img)
boxes, probs, points = mtcnn_net.detect(img, landmarks=True)
points = revise_points(boxes, points)
img_cropped = mtcnn_net(img, save_path="result.jpg")
left_eye_coord, right_eye_coord, nose_coord, mouth_coord, changed_pixels = draw_new_picture(points)
rate = changed_pixels / 160 / 160
print("rate = " + str(rate))

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))


# Or, if using for VGGFace2 classification
resnet.classify = True
sq_img = img_cropped.unsqueeze(0)
my_pic1 = rgb_to_img(np.array(sq_img.squeeze(0) * 255))

sq_img.requires_grad = True #for gradient calculation
img_probs = resnet(sq_img)

# print(img_probs)
tmp = img_probs.detach().numpy()[0].tolist()
print("Original result: " + str(tmp.index(max(tmp))+1) + "/" + str(len(tmp)))

# ########################################################
#FGSM ATTACK
img_probs.backward(torch.ones_like(img_probs))
fgsm_grad = sq_img.grad
i_num = 0
for i in sq_img.grad:
    j_num = 0
    for j in i:
        k_num = 0
        for k in j:
            l_num = 0
            for l in k:
                if l > 0:
                    fgsm_grad[i_num,j_num,k_num,l_num] = torch.tensor(1)
                elif l < 0:
                    fgsm_grad[i_num,j_num,k_num,l_num] = torch.tensor(-1)
                l_num += 1
            k_num += 1
        j_num += 1
    i_num += 1
        
# left_eye_coord = np.loadtxt('./mydata/left_eye_coord.txt', delimiter = ',')
# right_eye_coord = np.loadtxt('./mydata/right_eye_coord.txt', delimiter = ',')
# nose_coord = np.loadtxt('./mydata/nose_coord.txt', delimiter = ',')
# mouth_coord = np.loadtxt('./mydata/mouth_coord.txt', delimiter = ',')

def judge(coordx, coordy, targetcoord):
    if coordx > targetcoord[0] and coordx < targetcoord[2] and coordy > targetcoord[1] and coordy < targetcoord[3]:
        return True
    return False

def pixel_is_feature(coordx, coordy, left_eye_coord, right_eye_coord, nose_coord, mouth_coord):
    if judge(coordx, coordy, left_eye_coord) or judge(coordx, coordy, right_eye_coord) or judge(coordx, coordy, nose_coord) or judge(coordx, coordy, mouth_coord):
        return True
    return False
        

epslion = 0.001
original_pic = sq_img
while epslion < 0.01:
    print("epslion = " + str(epslion))
    sq_img = original_pic
    for i in range(0, 160):
        for j in range(0, 160):
            if pixel_is_feature(i, j, left_eye_coord, right_eye_coord, nose_coord, mouth_coord):
                sq_img[0][0][i][j] = sq_img[0][0][i][j] + fgsm_grad[0][0][i][j] * epslion
                sq_img[0][1][i][j] = sq_img[0][1][i][j] + fgsm_grad[0][1][i][j] * epslion
                sq_img[0][2][i][j] = sq_img[0][2][i][j] + fgsm_grad[0][2][i][j] * epslion
            
    img_probs = resnet(sq_img)
    tmp = img_probs.detach().numpy()[0].tolist()
    print("result considered features: " + str(tmp.index(max(tmp))+1) + "/" + str(len(tmp)))

    sq_img = original_pic
    sq_img = sq_img + fgsm_grad * epslion
    img_probs = resnet(sq_img)
    tmp = img_probs.detach().numpy()[0].tolist()
    print("result not considered features: " + str(tmp.index(max(tmp))+1) + "/" + str(len(tmp)))
    epslion += 0.001
# my_pic2 = sq_img.squeeze(0) * 255
# my_pic2 = rgb_to_img(np.array(my_pic2.detach()))


# #ADV_FACE中将e设为0.08
# #######################################################






# plt.figure()
# plt.subplot(1,2,1)
# plt.title('Before')
# plt.imshow(my_pic1)
# plt.subplot(1,2,2)
# plt.title('After, epslion = ' + str(epslion))
# plt.imshow(my_pic2)
# plt.show()
