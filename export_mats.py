# python export_mats.py './output/doctoring/ckpts/checkpoint-2018_08_28_2136_tcam_with_doctoring_lr0pt0001_outputSz256_margin0pt3-70841' 0
import tensorflow as tf
from classfile import NonTripletSet
import os, random, time
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow.contrib.slim as slim
from nets import resnet_v2
from scipy.io import savemat
import pickle
import sys
import glob
from shutil import copyfile

def main(pretrained_net, whichGPU):
    if not 'ilsvrc2012' in pretrained_net:
        iterStr = pretrained_net.split('-')[-1]
        splitStr = pretrained_net.split('/')
        outMatFolder = os.path.join('/'.join(splitStr[:np.where(np.array(splitStr)=='ckpts')[0][0]]),'results_small',iterStr)
    else:
        iterStr = 'ilsvrc2012'
        outMatFolder = os.path.join('./output/ilsvrc2012/results_small',iterStr,'mats')

    test_file = os.path.join('./input/test/small_test_by_hotel.txt')
    mean_file = os.path.join('./input/meanIm.npy')

    if not os.path.exists(outMatFolder):
        os.makedirs(outMatFolder)

    img_size = [256, 256]
    crop_size = [224, 224]
    batch_size = 100
    output_size = 256

    # Create test_data "batcher"
    #train_data = CombinatorialTripletSet(train_file, mean_file, img_size, crop_size, batch_size, num_pos_examples,isTraining=False)
    test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size,isTraining=False)

    image_batch = tf.placeholder(tf.float32, shape=[None, crop_size[0], crop_size[0], 3])

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=False)

    featLayer = 'resnet_v2_50/logits'
    non_norm_feat = tf.squeeze(layers[featLayer])
    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    conv1 = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_3/bottleneck_v2/add:0"))
    conv2 = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_4/bottleneck_v2/add:0"))
    conv3 = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_6/bottleneck_v2/add:0"))
    conv4 = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0"))
    weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/weights:0"))
    biases = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/biases:0"))
    gap = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"))

    ims_and_labels_path = os.path.join(outMatFolder,'ims_and_labels.pkl')
    if not os.path.exists(ims_and_labels_path):
        testingImsAndLabels = [(i,h) for h in test_data.hotels.keys() for i in test_data.hotels[h]['ims']]
        numTestingIms = len(testingImsAndLabels)
        with open(ims_and_labels_path, 'wb') as fp:
            pickle.dump(testingImsAndLabels, fp)
    else:
        with open (ims_and_labels_path, 'rb') as fp:
            testingImsAndLabels = pickle.load(fp)
        numTestingIms = len(testingImsAndLabels)

    print 'Num Images: ',numTestingIms

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list=whichGPU

    sess = tf.Session(config=c)
    saver = tf.train.Saver()

    saver.restore(sess, pretrained_net)

    testingFeats = np.empty((numTestingIms,output_size),dtype=np.float32)
    testingCV1 = np.empty((numTestingIms,conv1.shape[1]*conv1.shape[2],conv1.shape[3]),dtype=np.float32)
    testingCV2 = np.empty((numTestingIms,conv2.shape[1]*conv2.shape[2],conv2.shape[3]),dtype=np.float32)
    testingCV3 = np.empty((numTestingIms,conv3.shape[1]*conv3.shape[2],conv3.shape[3]),dtype=np.float32)
    testingCV4 = np.empty((numTestingIms,conv4.shape[1]*conv4.shape[2],conv4.shape[3]),dtype=np.float32)
    testingGAP = np.empty((numTestingIms,gap.shape[1]),dtype=np.float32)
    testingIms = np.empty((numTestingIms),dtype=object)
    testingLabels = np.empty((numTestingIms),dtype=np.int32)
    for idx in range(0,numTestingIms,batch_size):
        print idx, '/', numTestingIms
        il = testingImsAndLabels[idx:idx+batch_size]
        ims = [i[0] for i in il]
        labels = [i[1] for i in il]
        batch = test_data.getBatchFromImageList(ims)
        testingLabels[idx:idx+batch_size] = labels
        testingIms[idx:idx+batch_size] = ims
        ff, gg, cvOut1, cvOut2, cvOut3, cvOut4, wgts, bs = sess.run([non_norm_feat,gap,conv1,conv2,conv3,conv4,weights,biases], feed_dict={image_batch: batch, label_batch:labels})
        testingFeats[idx:idx+ff.shape[0],:] = np.squeeze(ff)
        testingGAP[idx:idx+ff.shape[0],:] = np.squeeze(gg)
        testingCV1[idx:idx+ff.shape[0],:,:] = cvOut1.reshape((cvOut1.shape[0],cvOut1.shape[1]*cvOut1.shape[2],cvOut1.shape[3]))
        testingCV2[idx:idx+ff.shape[0],:,:] = cvOut2.reshape((cvOut2.shape[0],cvOut2.shape[1]*cvOut2.shape[2],cvOut2.shape[3]))
        testingCV3[idx:idx+ff.shape[0],:,:] = cvOut3.reshape((cvOut3.shape[0],cvOut3.shape[1]*cvOut3.shape[2],cvOut3.shape[3]))
        testingCV4[idx:idx+ff.shape[0],:,:] = cvOut4.reshape((cvOut4.shape[0],cvOut4.shape[1]*cvOut4.shape[2],cvOut4.shape[3]))

    for cls in np.unique(testingLabels):
        inds = np.where(testingLabels==cls)[0]
        out_data = {}
        out_data['ims'] = testingIms[inds]
        out_data['labels'] = testingLabels[inds]
        out_data['feats'] = testingFeats[inds,:]
        out_data['gap'] = testingGAP[inds,:]
        out_data['conv1'] = testingCV1[inds,:,:]
        out_data['conv2'] = testingCV2[inds,:,:]
        out_data['conv3'] = testingCV3[inds,:,:]
        out_data['conv4'] = testingCV4[inds,:,:]
        out_data['weights'] = wgts
        out_data['biases'] = bs

        outfile = os.path.join(outMatFolder,str(cls)+'.mat')
        savemat(outfile,out_data)
        print outfile

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print 'Expected input parameters: pretrained_net,whichGPU'
    pretrained_net = args[1]
    whichGPU = args[2]
    main(pretrained_net,whichGPU)