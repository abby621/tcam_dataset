"""
# python same_chain_no_doctoring.py same_chain_margin diff_chain_margin batch_size output_size learning_rate whichGPU is_finetuning is_overfitting pretrained_net
# overfitting: python same_chain_no_doctoring.py .1 .3 120 256 .0001 2 False False None
# chop off last layer: python same_chain_no_doctoring.py .1 .3 120 256 .0001 2 True False './models/ilsvrc2012.ckpt'
# don't chop off last layer: python same_chain_no_doctoring.py 120 256 .0001 2 False False './models/ilsvrc2012.ckpt'
"""

import tensorflow as tf
from classfile import SameClassSet
import os.path
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import socket
import signal
import time
import sys
import itertools

def main(same_chain_margin,diff_chain_margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/sameChain/no_doctoring/ckpts'
    log_dir = './output/sameChain/no_doctoring/logs'
    train_filename = './input/train_by_chain.txt'
    mean_file = './input/meanIm.npy'

    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 1
    save_iters = 5000
    featLayer = 'resnet_v2_50/logits'

    is_training = True

    margin = float(diff_chain_margin)
    same_chain_margin = float(same_chain_margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    whichGPU = str(whichGPU)

    if batch_size%10 != 0:
        print 'Batch size must be divisible by 10!'
        sys.exit(0)

    num_pos_examples = batch_size/10

    # Create data "batcher"
    train_data = SameClassSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=is_training)

    if is_overfitting:
        good_chains = np.random.choice(train_data.chains.keys(),3,replace=False)
        for chain in train_data.chains.keys():
            if not chain in good_chains:
                train_data.chains.pop(chain)
            else:
                good_hotels = train_data.chains[chain].keys()[:num_pos_examples]
                for hotel in train_data.chains[chain].keys():
                    if not hotel in good_hotels:
                        train_data.chains[chain].pop(hotel)

    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_tcam_with_doctoring_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_margin'+str(margin).replace('.','pt')
    logfile_path = os.path.join(log_dir,param_str+'_train.txt')
    train_log_file = open(logfile_path,'a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print 'Margin: ',margin
    train_log_file.write('Margin: '+str(margin)+'\n')
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
    posIdx = np.floor(np.arange(0,batch_size)/num_pos_examples).astype('int')
    posIdx10 = num_pos_examples*posIdx
    posImInds = np.tile(posIdx10,(num_pos_examples,1)).transpose()+np.tile(np.arange(0,num_pos_examples),(batch_size,1))
    anchorInds = np.tile(np.arange(0,batch_size),(num_pos_examples,1)).transpose()
    posImInds_flat = posImInds.ravel()
    anchorInds_flat = anchorInds.ravel()
    posPairInds = zip(posImInds_flat,anchorInds_flat)
    ra, rb, rc = np.meshgrid(np.arange(0,batch_size),np.arange(0,batch_size),np.arange(0,num_pos_examples))
    bad_negatives = np.floor((ra)/num_pos_examples) == np.floor((rb)/num_pos_examples)
    bad_positives = np.mod(rb,num_pos_examples) == np.mod(rc,num_pos_examples)
    same_class_mask = ((1-bad_negatives)*(1-bad_positives)).astype('float32')

    chain_based_margin = np.zeros(same_class_mask.shape)
    chain_based_margin[:] = margin
    chain_based_margin[:batch_size/2,:batch_size/2,:] = same_chain_margin

    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    #D = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    D = 1 - tf.reduce_sum(tf.multiply(expanded_a, expanded_b), 2)

    posDists = tf.reshape(tf.gather_nd(D,posPairInds),(batch_size,num_pos_examples))
    shiftPosDists = tf.reshape(posDists,(1,batch_size,num_pos_examples))
    posDistsRep = tf.tile(shiftPosDists,(batch_size,1,1))
    allDists = tf.tile(tf.expand_dims(D,2),(1,1,num_pos_examples))

    all_loss = tf.maximum(0.,tf.multiply(same_class_mask,posDistsRep - allDists + chain_based_margin))
    non_zero_mask = tf.greater(all_loss, 0)
    non_zero_array = tf.boolean_mask(all_loss, non_zero_mask)
    loss = tf.reduce_mean(non_zero_array)

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

    restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
    restore_fn(sess)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time = time.time()
        batch, labels, ims = train_data.getBatch()
        people_masks = train_data.getPeopleMasks()
        batch_time = time.time() - start_time
        start_time = time.time()
        _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, people_mask_batch: people_masks})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.6f -- (batch creation: %.3f | training: %.3f sec)' % (step, loss_val,batch_time,duration)
        # print(out_str)
        if step % summary_iters == 0:
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
    if len(args) < 10:
        print 'Expected input parameters: same_chain_margin,diff_chain_margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net'
    same_chain_margin = args[1]
    diff_chain_margin = args[2]
    batch_size = args[3]
    output_size = args[4]
    learning_rate = args[5]
    whichGPU = args[6]
    is_finetuning = args[7]
    is_overfitting = args[8]
    pretrained_net = args[9]
    main(same_chain_margin,diff_chain_margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net)
