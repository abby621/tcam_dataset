import json

train_or_test = 'test'

jsonDataPath = './input/'+train_or_test+'_set.json'
with open(jsonDataPath) as f:
    data = json.load(f)

imDict = {}
for hotel in data:
    if train_or_test == 'train':
        for im in data[hotel]['expedia']:
            imFile = im.split('/')[-1]
            imDict[imFile] = {}
            imDict[imFile]['chainId'] = data[hotel]['chainId']
            imDict[imFile]['hotelId'] = hotel
        if 'tcam' in data[hotel].keys():
            for im in data[hotel]['tcam']:
                imFile = im.split('/')[-1]
                imDict[imFile] = {}
                imDict[imFile]['chainId'] = data[hotel]['chainId']
                imDict[imFile]['hotelId'] = hotel
    else:
        for im in data[hotel]['ims']:
            imFile = im.split('/')[-1]
            imDict[imFile] = {}
            imDict[imFile]['chainId'] = data[hotel]['chainId']
            imDict[imFile]['hotelId'] = hotel

import glob, os
dataset_path = '/project/focus/datasets/tcam_aaai/'+train_or_test+'/'
if train_or_test == 'train':
    files = glob.glob(os.path.join(dataset_path,'*/*/*.jpg'))
else:
    files = glob.glob(os.path.join(dataset_path,'*/*.jpg'))

# TODO: Break if len(files) > len(imDict.keys())

by_hotel = {}
by_chain = {}
for f in files:
    imFile = f.split('/')[-1]
    hotel = imDict[imFile]['hotelId']
    chain = imDict[imFile]['chainId']
    if not hotel in by_hotel.keys():
        by_hotel[hotel] = [f]
    else:
        by_hotel[hotel].append(f)
    if not chain in by_chain.keys():
        by_chain[chain] = [f]
    else:
        by_chain[chain].append(f)

with open('./input/'+train_or_test+'_by_hotel.txt', 'w') as f:
   for key in by_hotel.keys():
       f.writelines(' '.join(by_hotel[key]) + '\n')

with open('./input/'+train_or_test+'_by_chain.txt', 'w') as f:
   for key in by_chain.keys():
       f.writelines(' '.join(by_chain[key]) + '\n')
