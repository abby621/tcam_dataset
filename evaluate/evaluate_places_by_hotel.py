import tensorflow as tf
from classfile import *
import numpy as np
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import faiss
import h5py

def save_h5(data_description,data,data_type,path):
    h5_feats=h5py.File(path,'w')
    h5_feats.create_dataset(data_description, data=data, dtype=data_type)
    h5_feats.close()

pretrained_net = './models/places365.ckpt'
iterStr = 'places365'

output_dir = os.path.join('./output/doctoring/results',iterStr)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

whichGPU = 3
img_size = [256, 256]
crop_size = [224, 224]
batch_size = 120
output_size = 365
mean_file = './input/meanIm.npy'

train_dataset = './input/train_by_hotel.txt'
test_datasets = ['./input/test/by_hotel.txt','./input/occluded_test/by_hotel/0.txt','./input/occluded_test/by_hotel/1.txt','./input/occluded_test/by_hotel/2.txt','./input/occluded_test/by_hotel/3.txt']
test_names = ['by_hotel','occluded0','occluded1','occluded2','occluded3']

train_data = CombinatorialTripletSet(train_dataset, mean_file, img_size, crop_size, isTraining=False)
image_batch = tf.placeholder(tf.float32, shape=[None, crop_size[0], crop_size[0], 3])

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=False)

featLayer = 'resnet_v2_50/logits'
feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
c = tf.ConfigProto()
c.gpu_options.visible_device_list=str(whichGPU)
sess = tf.Session(config=c)
saver = tf.train.Saver()
saver.restore(sess, pretrained_net)

train_ims = []
train_classes = []
for ims,cls in zip(train_data.files,train_data.classes):
    for im in ims:
        train_ims.append(im)
        train_classes.append(int(cls))

train_ims = np.array(train_ims)
train_classes = np.array(train_classes)

train_index_dir = os.path.join(output_dir,'train.index')
if os.path.exists(train_index_dir):
    index2 = faiss.read_index(train_index_dir)
    print 'Loading index: ', train_index_dir
else:
    train_feats = np.zeros((train_ims.shape[0],output_size))
    for ix in range(0,train_ims.shape[0],batch_size):
        image_list = train_ims[ix:ix+batch_size]
        batch = train_data.getBatchFromImageList(image_list)
        ff = sess.run(feat,{image_batch:batch})
        train_feats[ix:ix+ff.shape[0],:] = ff
        print 'Train features: ', ix+ff.shape[0], ' out of ' , train_feats.shape[0]
    save_h5('train_ims',train_ims,h5py.special_dtype(vlen=bytes),os.path.join(output_dir,'trainIms.h5'))
    save_h5('train_classes',train_classes,'i8',os.path.join(output_dir,'trainClasses.h5'))
    save_h5('train_feats',train_feats,'f',os.path.join(output_dir,'trainFeats.h5'))
    index = faiss.IndexFlatIP(train_feats.shape[1])
    index2 = faiss.IndexIDMap(index)
    index2.add_with_ids(train_feats.astype('float32'),np.array(range(train_feats.shape[0]),dtype='int'))
    faiss.write_index(index2, train_index_dir)
    print 'Saved index: ',train_index_dir

for test_dataset, test_name in zip(test_datasets,test_names):
    test_output_dir = os.path.join(output_dir,test_name)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    test_data = CombinatorialTripletSet(test_dataset, mean_file, img_size, crop_size, isTraining=False)

    test_ims = []
    test_classes = []
    for ims,cls in zip(test_data.files,test_data.classes):
        for im in ims:
            test_ims.append(im)
            test_classes.append(int(cls))

    test_ims = np.array(test_ims)
    test_classes = np.array(test_classes)

    test_feats = np.zeros((test_ims.shape[0],output_size))
    for ix in range(0,test_ims.shape[0],batch_size):
        image_list = test_ims[ix:ix+batch_size]
        batch = test_data.getBatchFromImageList(image_list)
        ff = sess.run(feat,{image_batch:batch})
        test_feats[ix:ix+ff.shape[0],:] = ff
        print 'Test features: ',ix+ff.shape[0], ' out of ' , test_feats.shape[0]

    save_h5('test_ims',test_ims,h5py.special_dtype(vlen=bytes),os.path.join(test_output_dir,'testIms.h5'))
    save_h5('test_classes',test_classes,'i8',os.path.join(test_output_dir,'testClasses.h5'))
    save_h5('test_feats',test_feats,'f',os.path.join(test_output_dir,'testFeats.h5'))

    top_ims = np.zeros((test_feats.shape[0],100))
    for ind,feat,cls in zip(range(test_feats.shape[0]),test_feats,test_classes):
        result_dists, result_inds = index2.search(np.expand_dims(feat,0).astype('float32'),100)
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

    print test_output_dir
    print average_accuracy[0], average_accuracy[9], average_accuracy[99]
