import numpy as np
import pandas as pd
import argparse
import os
from utils.LoadDataset import getOneDataset,getTwoDataset
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset

import torchvision.transforms as transforms

from PIL import Image
from utils.evaluate import getAucScore
import warnings
from utils.evaluate import getAucScore
from utils.readfiles import getlabel,getDataset,getSecondDataset
import json
train_on_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
warnings.simplefilter("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--dataset_file",
        dest='dataset_file',
        default="/home/XXX/Music/dataset/jinnan2_round2_test_b_20190424",
        metavar="FILE",
        help="path to dataset file",
        type=str,
    )
    parser.add_argument(
        "--test_csv",
        dest='test_csv',
        default="/home/XXX/projects/dataset/tianchi_dataset/round2/tianchi_instances2_test_b_2019.csv",
        metavar="FILE",
        help="path to test csv",
        type=str,
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument(
        "--model",
        dest='model',
        default="/home/XXX/Music/vgg16_model_cls2_idx011/vgg_best_model.pkl",
        metavar="FILE",
        help="path to model",
        type=str,
    )

    return parser.parse_args()
import json
def main():
    args = parse_args()
    num_workers = 6
    print("loading data ...")

    valid_df = pd.read_csv(args.test_csv)
    valid_df.head()

    # y, le_full = valid_df['img_name'], len(valid_df['img_name'])
    # y = getlabel(y, le_full)
    data_transforms_valid = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(30, 60)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    valid_set = getDataset(datafolder=args.dataset_file, img_name=valid_df['img_name'], datatype='test', df=valid_df,
                           transform=data_transforms_valid)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, num_workers=num_workers,
                                               pin_memory=True)



    print("loading model ...")
    model_conv = torch.load(args.model)
    model_conv.eval()

    import csv
    all_pred = []
    all_imgname = []
    sumission_dict={}
    for batch_i, (data, target,img_name) in enumerate(valid_loader):
        data = data.cuda()
        output = model_conv(data)


        pred = output.sigmoid().cpu().data.numpy()
        all_pred.extend(pred.tolist())
        all_imgname.extend(img_name)
        # print(img_name)

    with open('/home/XXX/projects/models/test_b.csv', 'w') as csvfile:
        wri = csv.writer(csvfile, dialect='excel')
        wri.writerow(["img_name", "label", 'prob'])

        for i in range(len(all_pred)):
            wri.writerow([all_imgname[i], [round(all_pred[i][0], 2), round(all_pred[i][1], 2)]])
        csvfile.close()

    #     print("batch_i:", batch_i)
    #     for i in range(len(img_name)):
    #         predict_dict={}
    #         for j in range(pred.shape[1]):
    #             if j !=0:
    #                 predict_dict[str(j)]=float(pred[i][j])
    #         sumission_dict[img_name[i]]=predict_dict
    # sumission_file=open('/home/XXX/projects/tianchi/model_out/vgg16_with_normal/test_result.json','w')
    # json.dump(sumission_dict,sumission_file)
    pass
    #





if __name__ == "__main__":
    main()
