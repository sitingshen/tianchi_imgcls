import multiprocessing
from pycocotools.mask import *
import json
import numpy as np
import os

import json
import os
import argparse
import numpy as np
import pycocotools.mask as mask
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from pycocotools.mask import *
import json
import numpy as np
import os

bpath="/home/XXX/projects/models/test_ensemble/enesmble3_b/"

def arg_parse():
    parser = argparse.ArgumentParser(description='vis_bbox')
    parser.add_argument('--save_npy_root',default='npy07', type=str, help='save npy root')
    parser.add_argument('--binary_img_root', type=str,default='binary07', help='save binary img root')
    parser.add_argument('--result_path', type=str,default='segmentations_coco_tianchi_2019_test_b_ensemble2019-04-24_20-24-48.json', help='result_path')
    # parser.add_argument('--example_submission_path', type=str, help='example_submission_path')
    parser.add_argument('--test_info_path', type=str, default='/home/XXX/dataset/jinnan2_round2_test_b_20190424.json',help='test_info_path')
    parser.add_argument('--score_threshold', type=str, help='score_threshold', default=0.7)
    # parser.add_argument('--test_prob_root', type=str,default='/home/xiaolong/ss/projects/tianchi_dataset/round2/jinnan2_round2_train_20190401/tianchi_seg/img/test_probs/', help='test_prob_root')
    # parser.add_argument('--final_submit',default='submit07.json',type=str, help='save submit.json')
    return parser.parse_args()


def make_submit(image_name,preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit=dict()
    submit['image_name']= image_name
    submit['size']=(preds.shape[1],preds.shape[2])  #(height,width)
    submit['mask']=dict()

    for cls_id in range(0,5):      # 5 classes in this competition

        mask=preds[cls_id,:,:]
        cls_id_str=str(cls_id+1)   # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        rle = encode(fortran_mask) #encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        submit['mask'][cls_id_str]=rle

    return submit



def dump_2_json(submits,save_p):
    '''

    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    file = open(save_p, 'w', encoding='utf-8');
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()



def test2npy(save_npy_root, binary_img_root, result_path, file_test_data, score_threshold,dpi=100):

    file_result = open(result_path, 'r')
    file_result_data = json.load(file_result)

    qbar = tqdm(file_test_data)
    qbar.set_description('Test_set')

    for imgentry in qbar:

        img_npy_1=np.zeros([imgentry['height'],imgentry['width']],np.uint8)
        img_npy_2=np.zeros([imgentry['height'],imgentry['width']],np.uint8)
        img_npy_3=np.zeros([imgentry['height'],imgentry['width']],np.uint8)
        img_npy_4=np.zeros([imgentry['height'],imgentry['width']],np.uint8)
        img_npy_5=np.zeros([imgentry['height'],imgentry['width']],np.uint8)
        img_npy_list=[img_npy_1,img_npy_2,img_npy_3,img_npy_4,img_npy_5]
        for label_idx, labelentry in enumerate(file_result_data):
            if imgentry['id'] == labelentry['image_id']:
                if labelentry['score']<score_threshold:
                    continue
                binary_mask = mask.decode(labelentry['segmentation'])
                position = np.where(binary_mask == 1)

                img_npy_list[labelentry['category_id']-1][position[0],position[1]]=1

        for npy_idx,npy_entry in enumerate(img_npy_list):
            fig = plt.figure(frameon=False)
            fig.set_size_inches(imgentry['width'] / dpi, imgentry['height'] / dpi)
            # add some ax liked icon on the windows
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.axis('off')  # turn off the useless axis
            fig.add_axes(ax)
            # add our img into figure
            plt.imshow(npy_entry)
            # plt.show()
            img_save_path = os.path.join(binary_img_root,
                                         imgentry['file_name'][:-4] + '_' + str(npy_idx+1) + '.png')
            plt.savefig(img_save_path)
            plt.close()
            npy_save_path=os.path.join(save_npy_root,imgentry['file_name'][:-4]+'_'+str(npy_idx+1)+'.npy')
            np.save(npy_save_path,npy_entry)

    pass

def npy2submit(prediction_dir,json_p,test_json):

    with open(test_json,'r') as lf:
        loadjson = json.load(lf)

    submits_dict = dict()
    for image in loadjson['images']:

        image_name = image['file_name']
        img_id = image_name.split('.')
        preds = []
        for cls_id in range(1, 6):  # 5 classes in this competition
            cls_pred_name = "%s_%d.npy" % (img_id[0], cls_id)
            pred_p = os.path.join(prediction_dir, cls_pred_name)
            pred = np.load(pred_p)
            preds.append(pred)

        preds_np = np.array(preds)  # fake prediction
        submit = make_submit(image_name, preds_np)
        submits_dict[image_name] = submit

    dump_2_json(submits_dict, json_p)

if __name__ == '__main__':
    args = arg_parse()
    import time
    now = int(time.time())
    timearray = time.localtime(now)
    filetime = time.strftime("%Y-%m-%d_%H-%M-%S", timearray)


    # path like: save_root=r'E:\kaggle\tianchi\tianchi_result\round2\test\detectron_cascade_idx001\prediction'
    save_npy_root = bpath+args.save_npy_root
    # path like: save_root=r'E:\kaggle\tianchi\tianchi_result\round2\test\detectron_cascade_idx001\binary_mask'
    binary_img_root = bpath+args.binary_img_root
    # path like: annotation_json= r'E:\kaggle\tianchi\tianchi_result\round2\test\detectron_cascade_idx001\seg.json'
    result_path = bpath+args.result_path
    # path like: annotation_json= r'E:\kaggle\tianchi\jinnan2_round2_example_code_20190401\submit_example.json'
    # example_submission_path = args.example_submission_path
    # path like: annotation_json= r'E:\kaggle\tianchi\tianchi_instances2_test_a_2019.json'
    test_info_path = args.test_info_path
    # score_threshold like: score_threshold= 0.5
    score_threshold = args.score_threshold
    # path like: annotation_json= r'E:\kaggle\tianchi\tianchi_submit.json'
    final_submit = bpath+'submit07{}.json'.format(filetime)

    # test_prob_root=args.test_prob_root

    if not os.path.exists(binary_img_root):
        os.mkdir(binary_img_root)
    if not os.path.exists(save_npy_root):
        os.mkdir(save_npy_root)

    file_test = open(test_info_path, 'r')
    file_test_data = json.load(file_test)

    # test_a
    pool=multiprocessing.Pool(processes=10)
    result=[]

    for i in range(10):
        pool.apply_async(test2npy,(save_npy_root,binary_img_root,result_path, file_test_data['images'][i*150:(i+1)*150], score_threshold))

    #val
    # pool = multiprocessing.Pool(processes=5)
    # result = []
    #
    # for i in range(4):
    #     pool.apply_async(test2npy, (
    #     save_npy_root, binary_img_root, result_path, file_test_data['images'][i * 100:(i + 1) * 100], score_threshold))
    # pool.apply_async(test2npy, (
    #     save_npy_root, binary_img_root, result_path, file_test_data['images'][400:405], score_threshold))

    #mrnval
    # pool = multiprocessing.Pool(processes=10)
    # result = []
    #
    # for i in range(10):
    #     pool.apply_async(test2npy, (
    #         save_npy_root, binary_img_root, result_path, file_test_data['images'][i * 100:(i + 1) * 100],
    #         score_threshold))
    # pool.apply_async(test2npy, (
    #     save_npy_root, binary_img_root, result_path, file_test_data['images'][1000:1005], score_threshold))

    pool.close()
    pool.join()


    npy2submit(save_npy_root,final_submit,test_info_path)



