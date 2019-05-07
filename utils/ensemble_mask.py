import json
import csv
import argparse

rate=0.7

def arg_parse():
    parser = argparse.ArgumentParser(description='vis_bbox')
    parser.add_argument('--path_test_a_csv', type=str, help='path_test_a_csv',default='/home/XXX/projects/models/test_b.csv')
    parser.add_argument('--path_test_info_json', type=str, help='path_test_info_json',default='/home/XXX/Music/dataset/jinnan2_round2_test_b_20190424.json')
    parser.add_argument('--path_seg_final_result', type=str, help='path_seg_final_result',default='/home/XXX/files/mask_R50_cascade_idx017/test/coco_tianchi_test_b/generalized_rcnn/segmentations_coco_tianchi_test_b_results.json')
    parser.add_argument('--save_path', type=str, help='save_path', default='/home/XXX/files/mask_R50_cascade_idx017/test/coco_tianchi_test_b/ensemble07/')
    return parser.parse_args()

def ensemblemask(path_test_a_csv,path_test_info_json,path_seg_final_result,save_path):
    cls_bg=[]
    sub_bg=[]
    bg=[]
    iou=[]
    with open(path_test_a_csv, 'r') as csvfile:
        csv_results = csv.reader(csvfile)
        test_cls_imgname=[]
        test_cls_score=[]
        cls_em_sum=0
        for item in csv_results:
            if csv_results.line_num ==1 :
                continue
            test_cls_imgname.append(item[0])
            a = item[1].strip('[] ').split(',')
            test_cls_score.append([float(a[0]),float(a[1])])
            if float(a[0])>rate:
                cls_em_sum +=1
                cls_bg.append(item[0])

    with open(path_test_info_json,'r') as testf:
        test_info_json=json.load(testf)

    with open(path_seg_final_result,'r') as lf:
        final_result=json.load(lf)
        # final_result=copy.deepcopy(load)
        ensem=[]
        normidx = []
        for idx,cls_imgname in enumerate(test_cls_imgname):
            if test_cls_score[idx][0]>rate:

                img_id = -1
                for info_test in test_info_json['images']:
                    if cls_imgname==info_test['file_name']:
                        img_id=info_test['id']
                        break
                for sidx,seg in enumerate(final_result):
                    if seg['image_id']==img_id:
                        normidx.append(sidx)
        j=0
        normidx=sorted(normidx)
        print('add normal seg:',len(normidx))
        for sidx, seg in enumerate(final_result):
            if j<len(normidx) and sidx==normidx[j]:
                j+=1
            else:
                ensem.append(seg)
        save_file = open(save_path, 'w')
        json.dump(ensem, save_file)

        pass

import os
if __name__ == '__main__':
    args = arg_parse()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    import time

    now = int(time.time())
    timearray = time.localtime(now)
    filetime = time.strftime("%Y-%m-%d_%H-%M-%S", timearray)

    savepath=args.save_path+'segmentations_coco_tianchi_2019_test_b_ensemble{}.json'.format(filetime)
    ensemblemask(args.path_test_a_csv,args.path_test_info_json,args.path_seg_final_result,savepath)
