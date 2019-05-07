import json
import csv
import os


files=os.listdir('/home/XXX/dataset/jinnan2_round2_test_b_20190424')
files.sort(key=lambda x:int(x[:-4]))


with open('/home/XXX/projects/dataset/tianchi_dataset/round2/tianchi_instances2_test_b_2019.csv','w') as csvfile:
    wri = csv.writer(csvfile)
    wri.writerow(["img_name", "img_labels", "img_hotlabel","fblabel_list"])

    for file in files:
        wri.writerow((file,set(),[],[]))

    csvfile.close()

