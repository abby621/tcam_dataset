import numpy as np
import faiss
import h5py
import os

whichGPU = 2
pretrained_net = './output/expedia/ckpts/'
iterStr = pretrained_net.split('-')[-1]
output_dir = os.path.join('./output/expedia/results',iterStr)

def save_h5(data_description,data,data_type,path):
    h5_feats=h5py.File(path,'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()

def load_h5(data_description,path):
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data

res = faiss.StandardGpuResources()
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = whichGPU

train_feats = load_h5('train_feats',os.path.join(output_dir,'trainFeats.h5'))
train_classes = load_h5('train_classes',os.path.join(output_dir,'trainClasses.h5'))
train_ims = load_h5('train_ims',os.path.join(output_dir,'trainIms.h5'))
gpu_index = faiss.GpuIndexFlatIP(res, train_feats.shape[1],flat_config)
for feat in train_feats:
    gpu_index.add(np.expand_dims(feat,0))

test_datasets = ['./input/test_by_hotel.txt','./input/occluded_test/by_hotel/0.txt','./input/occluded_test/by_hotel/1.txt','./input/occluded_test/by_hotel/2.txt','./input/occluded_test/by_hotel/3.txt']
test_names = ['by_hotel','occluded0','occluded1','occluded2','occluded3']
for test_dataset, test_name in zip(test_datasets,test_names):
    test_output_dir = os.path.join(output_dir,test_name)
    if not os.path.exists(os.path.join(test_output_dir,'top_k.h5')):
        test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
        test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
        test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
        top_ims = np.zeros((test_feats.shape[0],100))
        for aa in range(0,test_feats.shape[0],100):
            ff = test_feats[aa:aa+100,:]
            result_dists, result_inds = gpu_index.search(ff.astype('float32'),100)
            top_ims[aa:aa+result_inds.shape[0],:] = result_inds
            # print aa
        correct_class = np.equal(np.tile(np.expand_dims(test_classes,1),100),train_classes[top_ims.astype('int')])
        top_k = np.zeros((test_classes.shape[0],100))
        for ind,rr in zip(range(correct_class.shape[0]),correct_class):
            if np.any(rr==True):
                top_k[ind,np.min(np.where(rr==True)[0]):] = 1
        average_accuracy = np.mean(top_k,axis=0)
        save_h5('top_ims',top_ims,'i8',os.path.join(test_output_dir,'top_ims.h5'))
        save_h5('top_k',top_k,'f',os.path.join(test_output_dir,'top_k.h5'))
        save_h5('average_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_accuracy.h5'))
        print iterStr, test_name, average_accuracy[0], average_accuracy[9], average_accuracy[99]

train_chain_file_path = './input/train_by_chain.txt'
with open(train_chain_file_path,'r') as f:
    train_chain_ims = f.readlines()

train_classes_by_chain = []
for cls in train_chain_ims:
    imsByClass = []
    for im in cls.split(' '):
        imsByClass.append(clsId)
        clsId = int(im.split('/')[3])
    train_classes_by_chain.append(np.unique(imsByClass))

jsonTestData = json.load(open('./input/test_set.json'))
jsonTrainData = json.load(open('./input/train_set.json'))

cls_to_chain = {}
for hotel in jsonTrainData.keys():
    if jsonTrainData[hotel]['chainId'] != -1:
        cls_to_chain[int(hotel)] = jsonTrainData[hotel]['chainId']

for hotel in jsonTestData.keys():
    if jsonTestData[hotel]['chainId'] != -1 and int(hotel) not in cls_to_chain.keys():
        cls_to_chain[int(hotel)] = jsonTestData[hotel]['chainId']

train_chain_classes = np.unique(np.concatenate(train_classes_by_chain).ravel())
by_chain_inds = np.where(np.in1d(train_classes,train_chain_classes)==True)[0]

del gpu_index

train_feats2 = train_feats[by_chain_inds,:]
train_classes2 = train_classes[by_chain_inds]
train_ims2 = train_ims[by_chain_inds]
gpu_index = faiss.GpuIndexFlatIP(res, train_feats2.shape[1],flat_config)
for feat in train_feats2:
    gpu_index.add(np.expand_dims(feat,0))

train_class_to_chain = np.array([cls_to_chain[cls] for cls in train_classes2])

test_datasets = ['./input/test_by_chain.txt','./input/occluded_test/by_chain/0.txt','./input/occluded_test/by_chain/1.txt','./input/occluded_test/by_chain/2.txt','./input/occluded_test/by_chain/3.txt']
test_names = ['by_chain','by_chain_occluded0','by_chain_occluded1','by_chain_occluded2','by_chain_occluded3']
for test_dataset, test_name in zip(test_datasets,test_names):
    test_output_dir = os.path.join(output_dir,test_name)
    if not os.path.exists(os.path.join(test_output_dir,'top_k.h5')):
        test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
        test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
        test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
        test_class_to_chain = np.array([cls_to_chain[cls] for cls in test_classes])
        top_ims = np.zeros((test_feats.shape[0],100))
        for aa in range(0,test_feats.shape[0],100):
            ff = test_feats[aa:aa+100,:]
            result_dists, result_inds = gpu_index.search(ff.astype('float32'),100)
            top_ims[aa:aa+result_inds.shape[0],:] = result_inds
        correct_class = np.equal(np.tile(np.expand_dims(test_class_to_chain,1),100),train_class_to_chain[top_ims.astype('int')])
        top_k = np.zeros((test_classes.shape[0],100))
        for ind,rr in zip(range(correct_class.shape[0]),correct_class):
            if np.any(rr==True):
                top_k[ind,np.min(np.where(rr==True)[0]):] = 1
        average_accuracy = np.mean(top_k,axis=0)
        save_h5('top_ims',top_ims,'i8',os.path.join(test_output_dir,'top_ims.h5'))
        save_h5('top_k',top_k,'f',os.path.join(test_output_dir,'top_k.h5'))
        save_h5('average_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_accuracy.h5'))
    else:
        average_accuracy = load_h5('average_accuracy',os.path.join(test_output_dir,'average_accuracy.h5'))
    print iterStr, test_name, average_accuracy[0], average_accuracy[9], average_accuracy[99]
