import numpy as np
import faiss
import h5py
import os

whichGPU = 3
pretrained_net = './output/sameChain/tcam/ckpts/checkpoint-2018_09_13_1336_tcam_with_doctoring_lr0pt0001_outputSz256_margin0pt3-19999'
iterStr = pretrained_net.split('-')[-1]
output_dir = os.path.join('./output/sameChain/results',iterStr)

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
    test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
    test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
    test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
    unique_classes = np.unique(train_classes)
    unique_classes_sorted = np.argsort(unique_classes)
    classification_scores = np.zeros((test_feats.shape[0],unique_classes.shape[0]))
    for aa in range(0,test_feats.shape[0],100):
        # print aa, ' out of ', test_feats.shape[0]
        ff = test_feats[aa:aa+100,:]
        result_dists, result_inds = gpu_index.search(ff.astype('float32'),1000)
        row_sums = result_dists.sum(axis=1)
        result_dists_normalized = result_dists / row_sums[:, np.newaxis]
        result_classes = train_classes[result_inds]
        resultInfo = [[unique_classes_sorted[np.searchsorted(unique_classes[unique_classes_sorted], np.unique(r,return_index=True)[0])],d[np.unique(r,return_index=True)[1]]] for r,d in zip(result_classes,result_dists_normalized)]
        for idx in range(len(resultInfo)):
            classification_scores[aa+idx,resultInfo[idx][0]] = resultInfo[idx][1]
    sorted_classes = np.zeros((test_feats.shape[0],unique_classes.shape[0]))
    for idx in range(test_feats.shape[0]):
        # print idx, ' out of ', test_feats.shape[0]
        sorted_classes[idx,:] = np.argsort(-classification_scores[idx])
    correct_cls_to_unique_ind = unique_classes_sorted[np.searchsorted(unique_classes[unique_classes_sorted], test_classes)]
    top_k = np.zeros((test_feats.shape[0],unique_classes.shape[0]))
    for idx in range(test_feats.shape[0]):
        topResult = np.where(sorted_classes[idx]==correct_cls_to_unique_ind[idx])[0][0]
        top_k[idx,topResult:] = 1
    average_accuracy = np.mean(top_k,axis=0)
    # save_h5('top_ims',top_ims,'i8',os.path.join(test_output_dir,'top_ims.h5'))
    save_h5('top_k',top_k,'f',os.path.join(test_output_dir,'top_k.h5'))
    save_h5('average_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_accuracy.h5'))
    print iterStr, test_name, average_accuracy[0], average_accuracy[9], average_accuracy[99]

import json
jsonTestData = json.load(open('./input/test_set.json'))
jsonTrainData = json.load(open('./input/train_set.json'))

cls_to_chain = {}
for hotel in jsonTrainData.keys():
    if jsonTrainData[hotel]['chainId'] != -1:
        cls_to_chain[int(hotel)] = jsonTrainData[hotel]['chainId']

for hotel in jsonTestData.keys():
    if jsonTestData[hotel]['chainId'] != -1 and int(hotel) not in cls_to_chain.keys():
        cls_to_chain[int(hotel)] = jsonTestData[hotel]['chainId']

by_chain_inds = np.where(np.in1d(train_classes,cls_to_chain.keys())==True)[0]

del gpu_index

train_feats2 = train_feats[by_chain_inds,:]
train_classes2 = train_classes[by_chain_inds]
train_ims2 = train_ims[by_chain_inds]

train_class_to_chain = np.array([cls_to_chain[cls] for cls in train_classes2])

gpu_index = faiss.GpuIndexFlatIP(res, train_feats2.shape[1],flat_config)
for feat in train_feats2:
    gpu_index.add(np.expand_dims(feat,0))

test_datasets = ['./input/test_by_chain.txt','./input/occluded_test/by_chain/0.txt','./input/occluded_test/by_chain/1.txt','./input/occluded_test/by_chain/2.txt','./input/occluded_test/by_chain/3.txt']
test_names = ['by_chain','by_chain_occluded0','by_chain_occluded1','by_chain_occluded2','by_chain_occluded3']
for test_dataset, test_name in zip(test_datasets,test_names):
    test_output_dir = os.path.join(output_dir,test_name)
    test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
    test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
    test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
    test_class_to_chain = np.array([cls_to_chain[cls] for cls in test_classes])
    unique_chains = np.unique(train_class_to_chain)
    unique_chains_sorted = np.argsort(unique_chains)
    chain_classification_scores = np.zeros((test_feats.shape[0],unique_chains.shape[0]))
    for aa in range(0,test_feats.shape[0],100):
        # print aa, ' out of ', test_feats.shape[0]
        ff = test_feats[aa:aa+100,:]
        result_dists, result_inds = gpu_index.search(ff.astype('float32'),1000)
        row_sums = result_dists.sum(axis=1)
        result_dists_normalized = result_dists / row_sums[:, np.newaxis]
        result_chains = train_class_to_chain[result_inds]
        resultInfo = [[unique_chains_sorted[np.searchsorted(unique_chains[unique_chains_sorted], np.unique(r,return_index=True)[0])],d[np.unique(r,return_index=True)[1]]] for r,d in zip(result_chains,result_dists_normalized)]
        for idx in range(len(resultInfo)):
            chain_classification_scores[aa+idx,resultInfo[idx][0]] = resultInfo[idx][1]
    sorted_chains = np.zeros((test_feats.shape[0],unique_chains.shape[0]))
    for idx in range(test_feats.shape[0]):
        # print idx, ' out of ', test_feats.shape[0]
        sorted_chains[idx,:] = np.argsort(-chain_classification_scores[idx])
    correct_chain_to_unique_ind = unique_chains_sorted[np.searchsorted(unique_chains[unique_chains_sorted], test_class_to_chain)]
    top_k = np.zeros((test_feats.shape[0],unique_classes.shape[0]))
    for idx in range(test_feats.shape[0]):
        # print idx, ' out of ', test_feats.shape[0]
        topResult = np.where(sorted_chains[idx].astype('int')==correct_chain_to_unique_ind[idx])[0][0]
        top_k[idx,topResult:] = 1
    average_accuracy = np.mean(top_k,axis=0)
    # save_h5('top_ims',top_ims,'i8',os.path.join(test_output_dir,'top_ims.h5'))
    save_h5('top_k',top_k,'f',os.path.join(test_output_dir,'top_k.h5'))
    save_h5('average_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_accuracy.h5'))
    print iterStr, test_name, average_accuracy[0], average_accuracy[2], average_accuracy[4], average_accuracy[9]
