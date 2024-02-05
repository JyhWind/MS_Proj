from models import mtcnn, inception_resnet_v1
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt


def rgb_to_img(img):
    r = img[0]  # 160x160
    g = img[1]  # 160x160
    b = img[2]  # 160x160

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

img = Image.open("1.jpg")

# Get cropped and prewhitened image tensor
img_cropped = mtcnn_net(img, save_path="result.png")

# Calculate embedding (unsqueeze to add batch dimension)
# img_embedding = resnet(img_cropped.unsqueeze(0))


# Or, if using for VGGFace2 classification
resnet.classify = True
sq_img = img_cropped.unsqueeze(0)
# print(sq_img.squeeze(0) * 256)
my_pic1 = rgb_to_img(np.array(sq_img.squeeze(0) * 128 + 127.5))


epslion = 0.5
print("epslion = " + str(epslion))
max_loop = 2
for loop in range(0, max_loop):

    sq_img.requires_grad = True  # for gradient calculation
    img_probs = resnet(sq_img)

    # print(img_probs.size())   # torch.Size([1, 8631])
    tmp = img_probs.detach().numpy()[0].tolist()
    if loop % 10 == 0:
        print(str(loop + 1) + "/" + str(max_loop))
        print("result: " + str(tmp.index(max(tmp))) + "/" + str(len(tmp)))

    # ########################################################
    # pgd ATTACK
    img_probs.backward(torch.ones_like(img_probs))

    pgd_grad = sq_img.grad  # torch.Size([1, 3, 160, 160]), 3通道160x160图片
    # print(pgd_grad.size())
    i_num = 0
    for i in sq_img.grad:
        j_num = 0
        for j in i:
            k_num = 0
            for k in j:
                l_num = 0
                for l in k:
                    if l > 0:
                        pgd_grad[i_num,j_num,k_num,l_num] = torch.tensor(1)
                    elif l < 0:
                        pgd_grad[i_num,j_num,k_num,l_num] = torch.tensor(-1)
                    l_num += 1
                k_num += 1
            j_num += 1
        i_num += 1
            
    project_value = 1
    if (epslion * torch.norm(pgd_grad)) > project_value:
        pgd_grad = pgd_grad * project_value / torch.norm(pgd_grad)



    sq_img = sq_img + pgd_grad * epslion
    sq_img = sq_img.detach() #important!



# print(sq_img.detach().squeeze(0) * 256)
my_pic = sq_img.squeeze(0) * 128 + 127.5
my_pic2 = rgb_to_img(np.array(my_pic.detach()))


# #ADV_FACE中将e设为0.08
# #######################################################

img_probs = resnet(sq_img)

# print(img_probs)
tmp = img_probs.detach().numpy()[0].tolist()
result_str = str(tmp.index(max(tmp))) + "/" + str(len(tmp))
print("result at last: " + result_str)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Before')
plt.imshow(my_pic1)
plt.subplot(1, 2, 2)
plt.title('After, result = ' + result_str)
plt.imshow(my_pic2)
plt.show()
