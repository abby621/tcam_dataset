import json

only_expedia = False
train_or_test = 'train'

jsonDataPath = './input/'+train_or_test+'_set.json'
with open(jsonDataPath) as f:
    data = json.load(f)

imDict = {}
for hotel in data:
    for source in data[hotel]['ims'].keys():
        if (not only_expedia) or (only_expedia and source=='expedia'):
            for im in data[hotel]['ims'][source].keys():
                imFile = data[hotel]['ims'][source][im]['path'].split('/')[-1]
                imDict[imFile] = {}
                imDict[imFile]['chainId'] = data[hotel]['chainId']
                imDict[imFile]['hotelId'] = hotel

import glob, os
dataset_path = './images/'+train_or_test+'/'
if train_or_test == 'train':
    files = glob.glob(os.path.join(dataset_path,'*/*/*.jpg'))
else:
    files = glob.glob(os.path.join(dataset_path,'*/*.jpg'))

# TODO: Break if len(files) > len(imDict.keys())

if train_or_test == 'train':
    numToInclude = 50000
else:
    numToInclude = 5000

by_hotel = {}
by_chain = {}
ctr = 0
while len(by_hotel.keys()) < numToInclude:
    print len(by_hotel.keys())
    f = files[ctr]
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
    ctr += 1

if only_expedia:
    by_hotel_path = './input/expedia_'+train_or_test+'_by_hotel.txt'
    by_chain_path = './input/expedia_'+train_or_test+'_by_chain.txt'
else:
    by_hotel_path = './input/'+train_or_test+'_by_hotel.txt'
    by_chain_path = './input/'+train_or_test+'_by_chain.txt'

with open(by_hotel_path, 'w') as f:
   for key in by_hotel.keys():
       f.writelines(' '.join(by_hotel[key]) + '\n')

with open(by_chain_path, 'w') as f:
   for key in by_chain.keys():
       f.writelines(' '.join(by_chain[key]) + '\n')
