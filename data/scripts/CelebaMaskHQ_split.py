import os
import json
import numpy as np

path = '/home/nsml/phydir/data'
id_list = np.loadtxt(os.path.join(path, 'ANNOTATION', 'identity_CelebA.txt'), dtype='str')
mapping_list = np.loadtxt(os.path.join(path, 'ORIGINAL', 'CelebAMask-HQ', 'CelebAMask-HQ-to-CelebA-mapping.txt'), dtype='str')

id_dict = {} # key: jpg, value: id
for row in id_list:
    id_dict[row[0]] = row[1]

mapping_dict = {} # key: CelebAMask-HQ idx, value: CelebA idx ( 0 - 119614.jpg)
for row in mapping_list:
    mapping_dict[row[0]] = row[2] #todo: check if it is correct, and first row is header


celeba_dict = {} # key: id, value: list of jpg

# classify id
data_path = os.path.join(path, 'ORIGINAL', 'CelebAMask-HQ', 'CelebA-HQ-img')
img_list = os.listdir(data_path)
for img in img_list:
    if img.endswith('.jpg'):
        # split name
        img_name = img.split('.')[0]
        # get id
        id = id_dict[mapping_dict[img_name]]
        # add to dict
        if id in celeba_dict.keys():
            celeba_dict[id].append(img)
        else:
            celeba_dict[id] = [img]

# split by phase
id_nums = len(celeba_dict.keys())
phase = {
    'train': int(id_nums * 20 / 24),
    'val': int(id_nums * 1 / 24),
    'test': int(id_nums * 3 / 24)
}

datalist = {
    'train': {},
    'val': {},
    'test': {}
}

# random select dir
for phase in phase.keys():
    for i in range(phase[phase]):
        # logger
        if i % 100 == 0:
            print(f'{phase} {i+1}/{phase[phase]}')

        # get id, list of jpg
        id, img_list = celeba_dict.popitem()

        # copy to target path
        for img in img_list:
            target_path = os.path.join(path, 'celebamask_hq', phase)
            os.system(f'cp {data_path}/{img} {target_path}/{img}')
        # add datalist
        if len(img_list) >= 6:
            datalist[phase][id] = img_list

# save datallist to json file
for phase in datalist.keys():
    with open(f'{path}/celebamask_hq/{phase}/datalist/combine.json', 'w') as f:
        json.dump(datalist[phase], f)