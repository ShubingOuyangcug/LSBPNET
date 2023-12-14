import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models import unetFEGcn
from utils.unet_dataset import read_tiff
from osgeo import gdal
from models import unet
from metrics import eval_metrics
from train import toString
import os
from metrics import eval_metrics
import numpy as np
import torch
from torchvision import transforms
from models import Mymodel3
from models import Mymodel3onenear
from models import Mymodel3onedistan
from models import Mymodel3max2
from models import Mymodelaverage2
from models import Mymodel3unetplusplus




def read_label(filename):
    dataset=gdal.Open(filename)    #打开文件
 
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数
 
    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset 
    return im_data

def eval(config,flag):
    device = torch.device('cuda:0')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    selected = config['train_model']['model'][config['train_model']['select']]
    if selected ==  'unetFEGcn':
        model = unetFEGcn.UNet(num_classes=config['num_classes'])
    if selected  == 'unet':
        model = unet.UNet(num_classes=config['num_classes'])
    if selected  == 'mymodelcoordconv':
        model = Mymodel3.MyModel(num_classes=config['num_classes'])
    if selected == 'Mymodel3onedistan':
        model = Mymodel3onedistan.MyModel(num_classes=config['num_classes'])
    if selected == 'Mymodelaverage2':
        model = Mymodelaverage2.MyModel(num_classes=config['num_classes'])
    if selected == 'Mymodel3max2':
        model = Mymodel3max2.MyModel(num_classes=config['num_classes'])
    if selected == 'Mymodel3onenear':
        model = Mymodel3onenear.MyModel(num_classes=config['num_classes'])
    if selected == 'Mymodel3unetplusplus':
        model = Mymodel3unetplusplus.MyModel(num_classes=config['num_classes'])


    selected = selected + flag

    check_point = os.path.join(config['save_model']['save_path'], selected+'_jx.pth')
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.15509938, 0.14781779, 0.16948901, 0.11077119, 0.43804327, 0.27022855, 0.09668206, -0.28062309],std=[0.0611714, 0.05668418, 0.05547993, 0.0539083, 0.11600339, 0.07835197, 0.03763121, 0.15423846])
        ]#-1.39e-05, -0.28062309,#,0.04194741, 0.15423846
    )
    model.load_state_dict(torch.load(check_point), False)
    model.cuda()
    model.eval()
    #混淆矩阵
    conf_matrix_test = np.zeros((config['num_classes'],config['num_classes']))

    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = 0.0
    unoin_sum = 0.0
    pixelAcc = 0.0
    mIoU = 0.0
        
    class_precision=np.zeros(config['num_classes'])
    class_recall=np.zeros(config['num_classes'])
    class_f1=np.zeros(config['num_classes'])
    with open(config['img_txt'], 'r', encoding='utf-8') as f:
        for line in f.readlines():

            image_name, n1, n2, n3, label_name = line.strip().split()



            root_dir = ''
            image_name = os.path.join(root_dir,image_name)
            label_name = os.path.join(root_dir,label_name)
            n1s_path = os.path.join(root_dir, n1)
            n2s_path = os.path.join(root_dir, n2)
            n3s_path = os.path.join(root_dir, n3)


            label = torch.from_numpy(np.asarray(read_label(label_name), dtype=np.int32)).long().cuda()

            image = read_tiff(image_name,train=True)
            image = np.array(image)
            image = np.transpose(image,(1,2,0))
            image = transforms.ToTensor()(image)
            image = image.to(torch.float32).cuda()
            x_range = torch.linspace(-1, 1, image.shape[-1])
            y, x = torch.meshgrid(x_range, x_range)
            coord_feat = torch.cat([x.reshape(1, 64, 64), y.reshape(1, 64, 64)], 0).cuda()
            image = transform(image).cuda()#加一维,batch_size=1
            # image = torch.cat([image, coord_feat], 0).cuda()
            image = image.unsqueeze(0)

            image1 = read_tiff(n1s_path,train=True)
            image1 = np.array(image1)
            image1 = np.transpose(image1, (1, 2, 0))
            image1 = transforms.ToTensor()(image1)
            image1 = image1.to(torch.float32).cuda()
            image1 = transform(image1).cuda()
            # image1 = torch.cat([image1, coord_feat], 0).cuda()
            image1 = image1.unsqueeze(0)

            image2 = read_tiff(n2s_path,train=True)
            image2 = np.array(image2)
            image2 = np.transpose(image2, (1, 2, 0))
            image2 = transforms.ToTensor()(image2)
            image2 = image2.to(torch.float32).cuda()
            image2 = transform(image2).cuda()
            # image2 = torch.cat([image2, coord_feat], 0).cuda()
            image2 = image2.unsqueeze(0)

            image3 = read_tiff(n3s_path,train=True)
            image3 = np.array(image3)
            image3 = np.transpose(image3, (1, 2, 0))
            image3 = transforms.ToTensor()(image3)
            image3 = image3.to(torch.float32).cuda()
            image3 = transform(image3).cuda()
            #image3 = torch.cat([image3, coord_feat], 0).cuda()
            image3 = image3.unsqueeze(0)

            # labelmulti
            image1label = torch.from_numpy(np.asarray(read_label(n1s_path.replace("image", "labelmuti")), dtype=np.int16)).long().cuda()
            image1label = image1label.unsqueeze(0)
            image2label = torch.from_numpy(np.asarray(read_label(n2s_path.replace("image", "labelmuti")), dtype=np.int16)).long().cuda()
            image2label = image2label.unsqueeze(0)
            image3label = torch.from_numpy(np.asarray(read_label(n3s_path.replace("image", "labelmuti")), dtype=np.int16)).long().cuda()
            image3label = image3label.unsqueeze(0)

            output = model(image,image1,image2,image3,image1label,image2label,image3label)
            # _, pred = output.max(1)
            # pred = pred.view(256, 256)
            # mask_im = pred.cpu().numpy().astype(np.uint8)
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(output, label, config['num_classes'],conf_matrix_test)
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)
                
            for i in range(config['num_classes']):
                #每一类的precision
                class_precision[i]=1.0*conf_matrix_test[i,i]/conf_matrix_test[:,i].sum()
                #每一类的recall
                class_recall[i]=1.0*conf_matrix_test[i,i]/conf_matrix_test[i].sum()
                #每一类的f1
                class_f1[i]=(2.0*class_precision[i]*class_recall[i])/(class_precision[i]+class_recall[i])
    print( 'OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(          
            pixelAcc, toString(mIoU), mIoU.mean(),toString(class_precision),toString(class_recall),toString(class_f1)))
    tttxxx = 'OA {:.5f} |IOU {} |mIoU {:.5f} |class_precision {}| class_recall {} | class_f1 {}|'.format(
            pixelAcc, toString(mIoU), mIoU.mean(),toString(class_precision),toString(class_recall),toString(class_f1))
    np.savetxt(os.path.join("confuse_matrix", selected+'_jx_matrix_test.txt'),conf_matrix_test,fmt="%d")

    with open(os.path.join("confuse_matrix", selected+'_jx_matrix_test.txt'), 'a') as file:
        file.write("\n")
        file.write(tttxxx)

if __name__ == "__main__":
    with open(r'eval_config3.json', encoding='utf-8') as f:
        config = json.load(f)
    flag=""
    eval(config,flag)