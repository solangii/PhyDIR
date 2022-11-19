import os
import json

# dir list
path = '/home/nsml/phydir/data/ORIGINAL/CASIA-WebFace/Small_Piece_For_Easy_Download/CASIA-WebFace'
dir_list = os.listdir(path)

nums = {
    'train': 4000,
    'val': 1000,
    'test': 1000
}

datalist = {
    'train': {},
    'val': {},
    'test': {}
}

# random select dir
for phase in nums.keys():
    for i in range(nums[phase]):
        # logger
        if i % 100 == 0:
            print(f'{phase} {i+1}/{nums[phase]}')

        dir_name = dir_list.pop()
        target_path = '/home/nsml/phydir/data/casia_webface'
        # file list
        file_list = os.listdir(os.path.join(path, dir_name))

        # add id directory to file list
        file_list = [os.path.join(dir_name, file_name) for file_name in file_list]

        # add datalist
        datalist[phase][dir_name] = file_list
        # os.system(f'cp -r {path}/{dir_name} {target_path}/{phase}/{dir_name}')

# save datallist to json file
for phase in datalist.keys():
    with open(f'/home/nsml/phydir/data/casia_webface/{phase}/datalist/combined.json', 'w') as f:
        json.dump(datalist[phase], f)