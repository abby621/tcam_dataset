"""
# python same_chain_no_doctoring.py fraction_same_chain batch_size output_size learning_rate whichGPU is_finetuning is_overfitting pretrained_net
# overfitting: python same_chain_no_doctoring_npairs.py .5 30 256 .0001 1 False True None
# chop off last layer: python same_chain_no_doctoring_npairs.py .5 120 256 .0001 2 True False './models/ilsvrc2012.ckpt'
# don't chop off last layer: python same_chain_no_doctoring_npairs.py .5 120 256 .0001 2 False False './models/ilsvrc2012.ckpt'
# don't chop off last layer + switch to more of the same chain: python same_chain_no_doctoring_npairs.py .75 120 256 .0001 2 False False './output/sameChain/no_doctoring/ckpts/checkpoint-2018_09_30_0809_lr1e-05_outputSz256_margin0pt4-38614'
"""

import tensorflow as tf
from classfile import SameChainNpairs
import os.path
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import socket
import signal
import time
import sys
import itertools
import json
from metric_learning import npairs_loss

def main(fraction_same_chain,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/sameChain/npairs/no_doctoring/ckpts'
    log_dir = './output/sameChain/npairs/no_doctoring/logs'
    train_filename = './input/train_by_hotel.txt'

    jsonTrainData = json.load(open('./input/train_set.json'))

    cls_to_chain = {}
    for hotel in jsonTrainData.keys():
        if jsonTrainData[hotel]['chainId'] != -1:
            cls_to_chain[int(hotel)] = jsonTrainData[hotel]['chainId']

    mean_file = './input/meanIm.npy'

    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 25
    save_iters = 5000
    featLayer = 'resnet_v2_50/logits'

    is_training = True

    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    whichGPU = str(whichGPU)
    fraction_same_chain = float(fraction_same_chain)

    if batch_size%10 != 0:
        print 'Batch size must be divisible by 10!'
        sys.exit(0)

    # Create data "batcher"
    train_data = SameChainNpairs(train_filename, cls_to_chain, mean_file, img_size, crop_size, batch_size, isTraining=is_training,fractionSameChain=fraction_same_chain)

    if is_overfitting.lower() == 'true':
        min_count = int(float(batch_size)*fraction_same_chain)
        good_chains1 = [c for c in train_data.chains.keys() if len(train_data.chains[c].keys()) > min_count]
        good_chains = np.random.choice(good_chains1,3,replace=False)
        for chain in train_data.chains.keys():
            if not chain in good_chains:
                train_data.chains.pop(chain)
            else:
                good_hotels = train_data.chains[chain].keys()[:min_count]
                for hotel in train_data.chains[chain].keys():
                    if not hotel in good_hotels:
                        train_data.chains[chain].pop(hotel)

    numChains = len(train_data.chains.keys())
    numHotels = np.sum([len(train_data.chains[c].keys()) for c in train_data.chains.keys()])
    numIms = np.sum([len(train_data.chains[c][h]['ims']) for c in train_data.chains.keys() for h in train_data.chains[c].keys()])

    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_fracSameChain'+str(fraction_same_chain).replace('.','pt')+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)
    logfile_path = os.path.join(log_dir,param_str+'_npairs_train.txt')
    train_log_file = open(logfile_path,'a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print 'Num chains:', numChains
    train_log_file.write('Num chains: '+str(numChains)+'\n')
    print 'Num hotels:', numHotels
    train_log_file.write('Num hotels: '+str(numHotels)+'\n')
    print 'Num ims:', numIms
    train_log_file.write('Num ims: '+str(numIms)+'\n')
    print 'Output size: ', output_size
    train_log_file.write('Output size: '+str(output_size)+'\n')
    print 'Learning rate: ',learning_rate
    train_log_file.write('Learning rate: '+str(learning_rate)+'\n')
    print 'Logging to: ',logfile_path
    train_log_file.write('Param_str: '+param_str+'\n')
    train_log_file.write('----------------\n')
    print ''
    print '------------'

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=[batch_size])
    noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
    final_batch = tf.add(image_batch,noise)

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=True)

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        if is_finetuning.lower() == 'true' and var.op.name.startswith('resnet_v2_50/logits') or 'momentum' in var.op.name.lower():
            excluded = True
        if not excluded:
            variables_to_restore.append(var)

    # numpy stuff for figuring out which elements are from the same class and which aren't
    anchor_inds = np.arange(0,batch_size,2)
    pos_inds = np.arange(1,batch_size,2)

    labels = tf.gather(label_batch,anchor_inds)

    all_feats = tf.squeeze(layers[featLayer])
    anchor_feats = tf.gather(all_feats,anchor_inds)
    pos_feats = tf.gather(all_feats,pos_inds)

    loss = npairs_loss(labels,anchor_feats,pos_feats)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)

    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=2000)

    # tf will consume any GPU it finds on the system. Following lines restrict it to specific gpus
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list=whichGPU

    print("Starting session...")
    sess = tf.Session(config=c)
    sess.run(init_op)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    if pretrained_net.lower() != 'none':
        restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
        restore_fn(sess)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time = time.time()
        batch, hotels, chains, ims = train_data.getBatch()
        batch_time = time.time() - start_time
        start_time = time.time()
        _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch,label_batch:hotels})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.6f (batch creation: %.3f | training: %.3f sec)' % (step, loss_val, batch_time,duration)
        # print(out_str)
        if step % summary_iters == 0 or is_overfitting.lower()=='true':
            print(out_str)
            train_log_file.write(out_str+'\n')
        # Update the events file.
        # summary_str = sess.run(summary_op)
        # writer.add_summary(summary_str, step)
        # writer.flush()
        #
        # Save a checkpoint
        if (step + 1) % save_iters == 0:
            print('Saving checkpoint at iteration: %d' % (step))
            pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        if (step + 1) == num_iters:
            print('Saving final')
            pretrained_net = os.path.join(ckpt_dir, 'final-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'final-',pretrained_net+'-'+str(step), ' saved!'

    sess.close()
    train_log_file.close()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 9:
        print 'Expected input parameters: fraction_same_chain, batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net'
    fraction_same_chain = args[1]
    batch_size = args[2]
    output_size = args[3]
    learning_rate = args[4]
    whichGPU = args[5]
    is_finetuning = args[6]
    is_overfitting = args[7]
    pretrained_net = args[8]
    main(fraction_same_chain,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net)
