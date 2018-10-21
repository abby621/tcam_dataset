"""
# ilsvrc: python evaluate/eval_resnet_feats_retrieval.py 0 ./models/ilsvrc2012.ckpt
# expedia: python evaluate/eval_resnet_feats_retrieval.py 0 ./output/expedia/ckpts/checkpoint-2018_09_19_1314_lr0pt0001_outputSz256_margin0pt3-74157
# no doctoring: python evaluate/eval_resnet_feats_retrieval.py 0 ./output/no_doctoring/ckpts/checkpoint-2018_09_19_0913_lr0pt0001_outputSz256_margin0pt3-75721
# doctoring: python evaluate/eval_resnet_feats_retrieval.py 0 ./output/doctoring/ckpts/checkpoint-2018_08_28_2136_tcam_with_doctoring_lr0pt0001_outputSz256_margin0pt3-70841
# chain: python evaluate/eval_resnet_feats_retrieval.py 0 ./output/sameChain/no_doctoring/ckpts/checkpoint-2018_09_30_0809_lr1e-05_outputSz256_margin0pt4-34999
# chain doctoring: python evaluate/eval_resnet_feats_retrieval.py 0 ./output/sameChain/doctoring/ckpts/checkpoint-2018_09_30_0854_lr0pt0001_outputSz256_margin0pt4-29999
"""

import numpy as np
import faiss
import h5py
import os
import sys
import numpy_indexed as npi

def save_h5(data_description,data,data_type,path):
    h5_feats=h5py.File(path,'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()

def load_h5(data_description,path):
    with h5py.File(path, 'r') as hf:
        data = hf[data_description][:]
    return data

def main(pretrained_net,whichGPU):
    if not 'ilsvrc2012' in pretrained_net:
        iterStr = pretrained_net.split('-')[-1]
        splitStr = pretrained_net.split('/')
        output_dir = os.path.join('/'.join(splitStr[:np.where(np.array(splitStr)=='ckpts')[0][0]]),'results',iterStr)
    else:
        iterStr = 'ilsvrc2012'
        output_dir = os.path.join('./output/ilsvrc2012/results',iterStr)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(whichGPU)

    train_feats = load_h5('train_feats',os.path.join(output_dir,'trainFeats.h5'))
    train_classes = load_h5('train_classes',os.path.join(output_dir,'trainClasses.h5'))
    train_ims = load_h5('train_ims',os.path.join(output_dir,'trainIms.h5'))
    gpu_index = faiss.GpuIndexFlatIP(res, train_feats.shape[1],flat_config)
    for feat in train_feats:
        gpu_index.add(np.expand_dims(feat,0))

    test_datasets = ['./input/test/test_by_hotel.txt','./input/occluded_test/by_hotel/0.txt','./input/occluded_test/by_hotel/1.txt','./input/occluded_test/by_hotel/2.txt','./input/occluded_test/by_hotel/3.txt']
    test_names = ['by_hotel','occluded0','occluded1','occluded2','occluded3']
    for test_dataset, test_name in zip(test_datasets,test_names):
        test_output_dir = os.path.join(output_dir,test_name)
        test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
        test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
        test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
        top_k = np.zeros((test_feats.shape[0],100))
        for aa in range(0,test_feats.shape[0],100):
            # print aa, ' out of ', test_feats.shape[0]
            ff = test_feats[aa:aa+100,:]
            result_dists, result_inds = gpu_index.search(ff.astype('float32'),1000)
            result_classes = train_classes[result_inds]
            for idx in range(ff.shape[0]):
                correctResults = np.where(result_classes[idx]==test_classes[aa+idx])[0]
                if len(correctResults) > 0 and correctResults[0] < 100:
                    topResult = correctResults[0]
                    top_k[aa+idx,topResult:] = 1
        average_accuracy = np.mean(top_k,axis=0)
        save_h5('average_retrieval_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_retrieval_accuracy.h5'))
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

    test_datasets = ['./input/test/test_by_chain.txt','./input/occluded_test/by_chain/0.txt','./input/occluded_test/by_chain/1.txt','./input/occluded_test/by_chain/2.txt','./input/occluded_test/by_chain/3.txt']
    test_names = ['by_chain','by_chain_occluded0','by_chain_occluded1','by_chain_occluded2','by_chain_occluded3']
    for test_dataset, test_name in zip(test_datasets,test_names):
        test_output_dir = os.path.join(output_dir,test_name)
        test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
        test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
        test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
        test_class_to_chain = np.array([cls_to_chain[cls] for cls in test_classes])
        top_k_chain = np.zeros((test_feats.shape[0],100))
        for aa in range(0,test_feats.shape[0],100):
            # print aa, ' out of ', test_feats.shape[0]
            ff = test_feats[aa:aa+100,:]
            result_dists, result_inds = gpu_index.search(ff.astype('float32'),1000)
            result_chains = train_class_to_chain[result_inds]
            for idx in range(ff.shape[0]):
                correctResults = np.where(result_chains[idx]==test_class_to_chain[aa+idx])[0]
                if len(correctResults) > 0 and correctResults[0] < 100:
                    topResult = correctResults[0]
                    top_k_chain[aa+idx,topResult:] = 1
        average_chain_accuracy = np.mean(top_k_chain,axis=0)
        save_h5('average_chain_retrieval_accuracy',average_chain_accuracy,'f',os.path.join(test_output_dir,'average_chain_retrieval_accuracy.h5'))
        print iterStr, test_name, average_chain_accuracy[0], average_chain_accuracy[2], average_chain_accuracy[4], average_chain_accuracy[9]

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print 'Expected input parameters:whichGPU,pretrained_net'
    whichGPU = args[1]
    pretrained_net = args[2]
    main(pretrained_net,whichGPU)
