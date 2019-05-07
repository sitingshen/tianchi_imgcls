import json
import os
import argparse
import matplotlib.pyplot as plt

def arg_parse():
    parser = argparse.ArgumentParser(description='create_nor_json')
    parser.add_argument('--img_root', type=str, default='', help='load image dir')
    parser.add_argument('--save_path', type=str, default='',help='save json path')
    parser.add_argument('--gt_annotation_json', type=str, default='',help='gt_anonotation_json path')
    return parser.parse_args()

def produce_dict(dataset_df, load_dict, img_root):
    dataset_dict = {}
    dataset_dict['info'] = load_dict['info']
    dataset_dict['licenses'] = load_dict['licenses']
    dataset_dict['categories'] = load_dict['categories']
    dataset_dict['images'] = []
    # #for normaltrain
    # dataset_dict['annotations'] = []
    for imgidx,imgentry in enumerate(dataset_df):
        imgpath=os.path.join(img_root,imgentry)
        img=plt.imread(imgpath,'r')
        img_height=img.shape[0]
        img_width=img.shape[1]
        images_newlist={}
        images_newlist['coco_url']=''
        images_newlist['data_captured']=''
        images_newlist['file_name']=imgentry
        images_newlist['flickr_url']=''
        images_newlist['id']=imgidx
        images_newlist['height']=img_height
        images_newlist['width']=img_width
        images_newlist['license']=1

        dataset_dict['images'].append(images_newlist)

        # annotations_list={}
        # annotations_list['id']=imgidx+len(load_dict['annotations'])
        # annotations_list['image_id']=imgidx+len(load_dict['images'])
        # annotations_list['category_id']=0
        # annotations_list['iscrowd']=0
        # annotations_list['segmentation']=[]
        # annotations_list['area']=1.0
        # annotations_list['bbox']=[0,0,int(img_width/3)-1,int(img_height/3)-1]
        # dataset_dict['annotations'].append(annotations_list)
        plt.close()
        print('image:{} wrote!'.format(imgentry))
    print('all images: {} are wrote!!!!!!'.format(len(dataset_df)))
    return dataset_dict


def produce_normal_json():
    args = arg_parse()

    # path like: img_root=r'E:\kaggle\tianchi\jinnan2_round1_train_20190222\normal'
    img_root = args.img_root
    # path like: save_root=r'E:\kaggle\tianchi\normal\path.json'
    save_path = args.save_path
    # path like: annotation_json= r'E:\kaggle\tianchi\jinnan2_round1_train_20190222\train_no_poly.json'
    gt_annotation_json = args.gt_annotation_json

    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)

    file_official = open(gt_annotation_json, 'r')
    file_official_data = json.load(file_official)

    dataset_df=os.listdir(img_root)

    normal_dict=produce_dict(dataset_df,file_official_data, img_root)
    save_file=open(save_path,'w')
    json.dump(normal_dict,save_file)
    save_file.close()
    pass


if __name__ == '__main__':
    produce_normal_json()