import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image

def write_parm(args,lr,step_size, gamma):
    fw=open(args.outputdir+'parm.txt','w')
    fw.write("dataset_file:"+args.dataset_file+"\n")
    fw.write("train_csv:" + args.train_csv + "\n")
    fw.write("batch_size:" + str(args.batch_size) + "\n")
    fw.write("epochs:" + str(args.epochs) + "\n")
    fw.write("img_size:" + str(args.img_size) + "\n")
    fw.write("lr:" + str(lr) + "\n")
    fw.write("step_size:" + str(step_size) + "\n")
    fw.write("gamma:" + str(gamma) + "\n")
    fw.close()


def getlabel(y,length,classifier=6):
    if classifier==2:
        for i in range(length):
            a = [0, 0]
            a[0] = int(y[i][1])
            a[1] = int(y[i][4])

            y[i] = a
    elif classifier==6:
        for i in range(length):
            a = [0, 0, 0, 0, 0,0]
            a[0] = int(y[i][1])
            a[1] = int(y[i][4])
            a[2] = int(y[i][7])
            a[3] = int(y[i][10])
            a[4] = int(y[i][13])
            a[5] = int(y[i][16])
            y[i]=a

    return torch.Tensor(y)

class getDataset(Dataset):
    def __init__(self, datafolder, img_name,datatype='train', df=None, transform=transforms.Compose([transforms.ToTensor()]),
                 y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train' or self.datatype == 'val':
            self.df = df.values
        self.image_files_list = img_name
        self.transform = transform

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        if self.datatype == 'train'  or self.datatype=='val':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]

        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((2,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test' or self.datatype=='val':
            # so that the images will be in a correct order
           return image, label, self.image_files_list[idx]

class getSecondDataset(Dataset):
    def __init__(self, train_datafolder, normal_datafolder, len_train, len_normal,
                 train_df=None, normal_df=None, transform=transforms.Compose([transforms.ToTensor()]),
                 train_y=None, normal_y=None):
        self.train_datafolder = train_datafolder
        self.normal_datafolder = normal_datafolder
        self.train_y = train_y
        self.normal_y = normal_y

        self.train_df = train_df.values
        self.normal_df = normal_df.values
        self.train_image_files_list = len_train
        self.normal_image_files_list = len_normal
        self.le_img= len_train + len_normal
        self.transform = transform

    def __len__(self):
        return self.le_img

    def __getitem__(self, idx):
        if idx < self.train_image_files_list:
            # print('train_image_files_list:idx:',idx)
            img_name = os.path.join(self.train_datafolder, self.train_df[idx][0])
            label = self.train_y[idx]

        else:
            # print('normal:idx:', idx)

            img_name = os.path.join(self.normal_datafolder,  self.normal_df[idx-self.train_image_files_list][0])
            label = self.normal_y[idx-self.train_image_files_list]

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        return image, label
