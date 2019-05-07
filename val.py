import numpy as np
import pandas as pd
import argparse
import os
from utils.LoadDataset import getOneDataset,getTwoDataset
import torch
import warnings
from utils.evaluate import getAucScore,getFlattenAucScore
from utils.readfiles import getlabel,getDataset
from sklearn.metrics import precision_score,recall_score

train_on_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
warnings.simplefilter("ignore", category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--dataset_file",
        dest='dataset_file',
        default="",
        metavar="FILE",
        help="path to dataset file",
        type=str,
    )
    parser.add_argument(
        "--val_csv",
        dest='val_csv',
        default="",
        metavar="FILE",
        help="path to test csv",
        type=str,
    )
    # parser.add_argument(
    #     "--normal_file",
    #     dest='normal_file',
    #     default="/home/XXX/projects/tianchi_dataset/round1/jinnan2_round1_train_20190222/normal/",
    #     metavar="FILE",
    #     help="path to normal file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "--normal_csv",
    #     dest='normal_csv',
    #     default="/home/XXX/projects/tianchi_dataset/round1/jinnan2_round1_train_20190305/split/fbrate/2_3/tianchi_instances_rnval2019.csv",
    #     metavar="FILE",
    #     help="path to normal csv",
    #     type=str,
    # )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument(
        "--model",
        dest='model',
        default="/home/XXX/projects/models/vgg16_model_cls6_round2/vgg_model_200.pkl",
        metavar="FILE",
        help="path to model",
        type=str,
    )

    return parser.parse_args()

def val_main(model_conv=None):
    args = parse_args()
    num_workers = 6
    print("loading data ...")

    valid_loader = getOneDataset(args,'val',num_workers,size=[224,224],classifier=6)

    print("loading model ...")
    if model_conv==None:
        model_conv = torch.load(args.model)
    model_conv.eval()
    all_target=[]
    all_pred=[]
    all_imgname=[]
    import csv

    for batch_i, (data, target,imgname) in enumerate(valid_loader):
        data = data.cuda()
        output = model_conv(data)
        gt = target.cpu().data.numpy()
        pred =  output.sigmoid().cpu().data.numpy()
        all_target.extend(np.array(gt).tolist())
        all_pred.extend(pred.tolist())
        all_imgname.extend(imgname)



    with open('/home/XXX/projects/models/vgg16_model_cls6_round2/valrn_round2_200_5.csv', 'w') as csvfile:
        wri = csv.writer(csvfile, dialect='excel')
        wri.writerow(["img_name",'prob'])


        for i in range(len(all_pred)):
            # wri.writerow([all_imgname[i], [1,0] if np.argmax(all_pred[i])==0 else [0,1] , [round(all_pred[i][0],2),round(all_pred[i][1],2)]])
            wri.writerow([all_imgname[i], [round(all_pred[i][0], 5), round(all_pred[i][1], 5), round(all_pred[i][2], 5), round(all_pred[i][3], 5), round(all_pred[i][4], 5), round(all_pred[i][5], 5)]])
        csvfile.close()

        auc=getFlattenAucScore(np.array(all_pred),np.array(all_target))
        print('##########################')
        print("val auc:",auc)
        print('##########################')
        return auc

    auc = getFlattenAucScore(np.array(all_pred), np.array(all_target))
    print('##########################')
    print("val auc:", auc)
    print('##########################')

    all_pred = np.array(all_pred)
    all_pred[all_pred >= 0.9] = 1
    all_pred[all_pred < 0.9] = 0
    print('precision:', precision_score(np.array(all_target), all_pred, average='micro'))
    print('recall:', recall_score(np.array(all_target), all_pred, average='micro'))
    return auc

if __name__ == "__main__":
    val_main()
