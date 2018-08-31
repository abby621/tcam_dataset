import numpy as np
import faiss
import h5py
import os

whichGPU = 3
pretrained_net = './models/ilsvrc2012.ckpt'
iterStr = 'ilsvrc2012'

output_dir = os.path.join('./output/ilsvrc2012/results',iterStr)

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

test_datasets = ['./input/test_by_hotel.txt','./input/occluded_test/by_hotel/0.txt','./input/occluded_test/by_hotel/1.txt','./input/occluded_test/by_hotel/2.txt','./input/occluded_test/by_hotel/3.txt','./input/test_by_chain.txt','./input/occluded_test/by_chain/0.txt','./input/occluded_test/by_chain/1.txt','./input/occluded_test/by_chain/2.txt','./input/occluded_test/by_chain/3.txt']
test_names = ['by_hotel','occluded0','occluded1','occluded2','occluded3','by_chain','by_chain_occluded0','by_chain_occluded1','by_chain_occluded2','by_chain_occluded3']
for test_dataset, test_name in zip(test_datasets,test_names):
    test_output_dir = os.path.join(output_dir,test_name)
    print test_output_dir
    if not os.path.exists(os.path.join(test_output_dir,'top_k.h5')):
        test_feats = load_h5('test_feats',os.path.join(test_output_dir,'testFeats.h5'))
        test_ims = load_h5('test_ims',os.path.join(test_output_dir,'testIms.h5'))
        test_classes = load_h5('test_classes',os.path.join(test_output_dir,'testClasses.h5'))
        top_ims = np.zeros((test_feats.shape[0],100))
        for ind,feat,cls in zip(range(test_feats.shape[0]),test_feats,test_classes):
            result_dists, result_inds = gpu_index.search(np.expand_dims(feat,0).astype('float32'),100)
            top_ims[ind,:] = result_inds
        correct_class = np.equal(np.tile(np.expand_dims(test_classes,1),100),train_classes[top_ims.astype('int')])
        top_k = np.zeros((test_classes.shape[0],100))
        for ind,rr in zip(range(correct_class.shape[0]),correct_class):
            if np.any(rr==True):
                top_k[ind,np.min(np.where(rr==True)[0]):] = 1
        average_accuracy = np.mean(top_k,axis=0)
        save_h5('top_ims',top_ims,'i8',os.path.join(test_output_dir,'top_ims.h5'))
        save_h5('top_k',top_k,'f',os.path.join(test_output_dir,'top_k.h5'))
        save_h5('average_accuracy',average_accuracy,'f',os.path.join(test_output_dir,'average_accuracy.h5'))
        print average_accuracy[0], average_accuracy[9], average_accuracy[99]
