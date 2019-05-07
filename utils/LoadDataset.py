import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from utils.readfiles import getlabel,getDataset,getSecondDataset



def getTestDataset(args,num_workers):


    df = pd.read_csv(args.test_csv)
    df.head()

    y, le_full = df['img_hotlabel'], len(df['img_hotlabel'])
    y = getlabel(y, le_full)
    data_transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_set = getDataset(datafolder=args.dataset_file, img_name=df['img_name'], datatype='test',
                           df=df, transform=data_transforms_valid, y=y)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader

def getOneDataset(args,mold,num_workers,size=[224,224],classifier=6):

    if mold == 'train':
        df = pd.read_csv(args.train_csv)
        df.head()
    elif mold == 'val':
        df = pd.read_csv(args.val_csv)
        df.head()
    elif mold == 'test':
        df = pd.read_csv(args.test_csv)
        df.head()


    y, le_full = df['img_hotlabel'], len(df['img_hotlabel'])
    y = getlabel(y, le_full,classifier)
    data_transforms_valid = transforms.Compose([
        transforms.Resize((size[0], size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(30,60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



    data_set = getDataset(datafolder=args.dataset_file, img_name=df['img_name'], datatype=mold,
                           df=df, transform=data_transforms_valid, y=y)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader



def getTwoDataset(args,mold,num_workers):

    if mold == 'train':
        df = pd.read_csv(args.train_csv)
        df.head()

        normal_df = pd.read_csv(args.normal_csv)
        normal_df.head()


    elif mold == 'val':
        df = pd.read_csv(args.val_csv)
        df.head()
        normal_df = pd.read_csv(args.normal_csv)
        normal_df.head()

    y, le_full = df['img_hotlabel'], len(df['img_hotlabel'])
    y = getlabel(y, le_full)
    y_normal, le_full_normal = normal_df['img_hotlabel'], len(normal_df['img_hotlabel'])
    y_normal = getlabel(y_normal, le_full_normal)

    data_transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    data_set = getSecondDataset(args.dataset_file, args.normal_file, le_full, le_full_normal,
                                 train_df=df,
                                 normal_df=normal_df, transform=data_transforms_valid, train_y=y,
                                 normal_y=y_normal)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, num_workers=num_workers,
                                               pin_memory=True)
    return data_loader


