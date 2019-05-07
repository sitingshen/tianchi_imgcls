import argparse
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
import warnings
from utils.evaluate import getAucScore,getFlattenAucScore
from utils.readfiles import getlabel,getDataset,write_parm
from tensorboardX import SummaryWriter
from utils.LoadDataset import getOneDataset,getTwoDataset
from model.net import Net

train_on_gpu = True

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
        "--train_csv",
        dest='train_csv',
        default="",
        metavar="FILE",
        help="path to train csv",
        type=str,
    )
    parser.add_argument(
        "--normal_file",
        dest='normal_file',
        default="",
        metavar="FILE",
        help="path to normal file",
        type=str,
    )
    parser.add_argument(
        "--normal_csv",
        dest='normal_csv',
        default="",
        metavar="FILE",
        help="path to normal csv",
        type=str,
    )

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--img_size", type=int, default=[224,224])


    parser.add_argument(
        "--outputdir",
        dest='outputdir',
        default="",
        metavar="FILE",
        help="path to train csv",
        type=str,
    )

    return parser.parse_args()
from val import val_main

def main():
    args=parse_args()
    lr = 0.01
    step_size = 350
    gamma = 0.1
    num_workers = 6
    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)
    write_parm(args, lr, step_size, gamma)

    # train_loader=getTwoDataset(args,'train',num_workers)
    train_loader=getOneDataset(args,'train',num_workers,args.img_size,classifier=2)

    model=Net()
    # model_dict=model.state_dict()

    # shuffle=ShuffleNetV2_back()
    # shuffle.load_state_dict(torch.load('/home/xiaolong/ss/projects/Shufflenet-v2-Pytorch-master/shufflenetv2_x1_69.402_88.374.pth.tar'))
    # shuffle_dict=shuffle.state_dict()
    # shuffle_dict={k:v for k,v in shuffle_dict.items() if k in model_dict}
    # model_dict.update(shuffle_dict)
    # model.load_state_dict(model_dict)

    # model=Net()
    # model=torch.load('/home/xiaolong/ss/projects/tianchi/model_out/Vgg16/vgg_model_200.pkl')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,min_lr=0.00001, verbose=True)

    loss_list = []
    all_target = []
    all_pred = []

    model.train()
    max_auc=0

    writer = SummaryWriter(log_dir="../log")
    for epoch in range(0, args.epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = []

        for batch_i, (data, target) in enumerate(train_loader):

            output = model(data)

            all_target.extend(np.array(target).tolist())
            all_pred.extend(output.tolist())

            loss = criterion(output, target)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Loss:{}'.format(loss))
            writer.add_scalar("loss", loss, epoch)


        loss_list.append(np.mean(train_loss))
        exp_lr_scheduler.step(np.mean(train_loss))

        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}')

        if epoch % 20 == 0:
            auc=getFlattenAucScore(np.array(all_pred),np.array(all_target))
            writer.add_scalar("train_auc", auc, epoch)
            print("train_auc:",auc)
            val_auc=val_main(model)
            if max_auc<val_auc:
                max_auc=val_auc
                torch.save(model, args.outputdir + 'best_model.pkl')
                print('save model best_model eporch:' + str(epoch))

        print("-----------------------------------------------------------------")
        if epoch % 50 == 0:
            val_main(model)
            torch.save(model,args.outputdir+'model_' + str(epoch) + '.pkl')
            print('save model model_' + str(epoch) + '.pkl')


if __name__ == "__main__":
    main()