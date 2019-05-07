import json
import csv
import os
import matplotlib.pyplot as plt
cls_bg=[]
sub_bg=[]
bg=[]
iou=[]
with open('/home/XXX/projects/tianchi/model_out/vgg16_model_cls2/test_b.csv', 'r') as csvfile:
    csv_results = csv.reader(csvfile)
    test_cls_imgname=[]
    test_cls_score=[]
    cls_em_sum=0
    for item in csv_results:
        if csv_results.line_num ==1 :
            continue
        test_cls_imgname.append(item[0])
        a = item[2].strip('[] ').split(',')
        test_cls_score.append([float(a[0]),float(a[1])])
        if float(a[0])>0.96:
            cls_em_sum +=1
            cls_bg.append(item[0])


with open("/home/XXX/projects/tianchi_dataset/round1/result/submission.json",'r') as lf:
    load=json.load(lf)
    em_sum=0
    for idx, entry in enumerate(load['results']):
        if load['results'][idx]['rects']==[]:
            em_sum+=1
            sub_bg.append(load['results'][idx]['filename'])

    for idx,entry in enumerate(load['results']):

        if entry['filename']==test_cls_imgname[idx]:
            if test_cls_score[idx][0]>0.96:
                load['results'][idx]['rects']=[]

    save_file = open('/home/XXX/projects/tianchi/gettest/model_score/cascade/submit_test_b.json', 'w')
    json.dump(load, save_file)
    print('ensemble submission!cls_em_sum normal:{}  em_sum normal:{}'.format(cls_em_sum,em_sum))




