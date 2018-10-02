"""
# python same_chain_doctoring.py fraction_same_chain same_chain_margin diff_chain_margin batch_size output_size learning_rate whichGPU is_finetuning is_overfitting pretrained_net
# overfitting: python same_chain_doctoring.py .5 .2 .4 120 256 .0001 1 False True None
# chop off last layer: python same_chain_doctoring.py .5 .2 .4 120 256 .0001 1 True False './models/ilsvrc2012.ckpt'
# don't chop off last layer: python same_chain_doctoring.py .5 .2 .4 120 256 .0001 1 False False './models/ilsvrc2012.ckpt'
# don't chop off last layer + more of the same chain: python same_chain_doctoring.py .75 .3 .5 120 256 .0001 1 False False './output/sameChain/no_doctoring/ckpts/checkpoint-2018_09_30_0809_lr1e-05_outputSz256_margin0pt4-38614'
"""

import tensorflow as tf
from classfile import SameChainSet
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
import json

def main(fraction_same_chain,same_chain_margin,diff_chain_margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/sameChain/doctoring/ckpts'
    log_dir = './output/sameChain/doctoring/logs'
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
    summary_iters = 1
    save_iters = 5000
    featLayer = 'resnet_v2_50/logits'

    is_training = True

    margin = float(diff_chain_margin)
    same_chain_margin = float(same_chain_margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    fraction_same_chain = float(fraction_same_chain)
    whichGPU = str(whichGPU)

    if batch_size%10 != 0:
        print 'Batch size must be divisible by 10!'
        sys.exit(0)

    num_pos_examples = batch_size/10

    # Create data "batcher"
    train_data = SameChainSet(train_filename, cls_to_chain, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=is_training,fractionSameChain=fraction_same_chain)

    if is_overfitting.lower() == 'true':
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
    param_str = datestr+'_fracSameChain'+str(fraction_same_chain).replace('.','pt')+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_margin'+str(margin).replace('.','pt')
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
    people_mask_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 1])

    # doctor image params
    percent_crop = .5
    percent_people = .5
    percent_rotate = .2
    percent_filters = .4
    percent_text = .1

    # # richard's argument: since the data is randomly loaded, we don't need to change the indices that we perform operations on every time; i am on board with this, but had already implemented the random crops, so will leave that for now
    # # apply random rotations
    num_rotate = int(batch_size*percent_rotate)
    rotate_inds = np.random.choice(np.arange(0,batch_size),num_rotate,replace=False)
    rotate_vals = np.random.randint(-65,65,num_rotate).astype('float32')/float(100)
    rotate_angles = np.zeros((batch_size))
    rotate_angles[rotate_inds] = rotate_vals
    rotated_batch = tf.contrib.image.rotate(image_batch,rotate_angles,interpolation='BILINEAR')

    # do random crops
    num_to_crop = int(batch_size*percent_crop)
    num_to_not_crop = batch_size - num_to_crop

    shuffled_inds = tf.random_shuffle(np.arange(0,batch_size,dtype='int32'))
    # shuffled_inds = np.arange(0,batch_size,dtype='int32')
    # np.random.shuffle(shuffled_inds)
    crop_inds = tf.slice(shuffled_inds,[0],[num_to_crop])
    uncropped_inds = tf.slice(shuffled_inds,[num_to_crop],[num_to_not_crop])

    # crop_ratio = float(3)/float(5)
    # crop_yx = tf.random_uniform([num_to_crop,2], 0,1-crop_ratio, dtype=tf.float32, seed=0)
    # crop_sz = tf.add(crop_yx,np.tile([crop_ratio,crop_ratio],[num_to_crop, 1]))
    # crop_boxes = tf.concat([crop_yx,crop_sz],axis=1)

    # randomly select a crop between 3/5 of the image and the entire image
    crop_ratio = tf.random_uniform([num_to_crop,1], float(3)/float(5), 1, dtype=tf.float32, seed=0)
    # randomly select a starting location between 0 and the max valid x position
    crop_yx = tf.random_uniform([1,2],0.,1.-crop_ratio, dtype=tf.float32,seed=0)
    crop_sz = tf.add(crop_yx,tf.concat([crop_ratio,crop_ratio],axis=1))
    crop_boxes = tf.concat([crop_yx,crop_sz],axis=1)

    uncropped_boxes = np.tile([0,0,1,1],[num_to_not_crop,1])

    all_inds = tf.concat([crop_inds,uncropped_inds],axis=0)
    all_boxes = tf.concat([crop_boxes,uncropped_boxes],axis=0)

    sorted_inds = tf.nn.top_k(-shuffled_inds,sorted=True,k=batch_size).indices
    cropped_batch = tf.gather(tf.image.crop_and_resize(rotated_batch,all_boxes,all_inds,crop_size),sorted_inds)

    # apply different filters
    flt_image = convert_image_dtype(cropped_batch, dtypes.float32)

    num_to_filter = int(batch_size*percent_filters)

    filter_inds = np.random.choice(np.arange(0,batch_size),num_to_filter,replace=False)
    filter_mask = np.zeros(batch_size)
    filter_mask[filter_inds] = 1
    filter_mask = filter_mask.astype('float32')
    inv_filter_mask = np.ones(batch_size)
    inv_filter_mask[filter_inds] = 0
    inv_filter_mask = inv_filter_mask.astype('float32')

    #
    hsv = gen_image_ops.rgb_to_hsv(flt_image)
    hue = array_ops.slice(hsv, [0, 0, 0, 0], [batch_size, -1, -1, 1])
    saturation = array_ops.slice(hsv, [0, 0, 0, 1], [batch_size, -1, -1, 1])
    value = array_ops.slice(hsv, [0, 0, 0, 2], [batch_size, -1, -1, 1])

    # hue
    delta_vals = random_ops.random_uniform([batch_size],-.15,.15)
    hue_deltas = tf.multiply(filter_mask,delta_vals)
    hue_deltas2 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(hue_deltas,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    # hue = math_ops.mod(hue + (hue_deltas2 + 1.), 1.)
    hue_mod = tf.add(hue,hue_deltas2)
    hue = clip_ops.clip_by_value(hue_mod,0.0,1.0)

    # saturation
    saturation_factor = random_ops.random_uniform([batch_size],-.05,.05)
    saturation_factor2 = tf.multiply(filter_mask,saturation_factor)
    saturation_factor3 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(saturation_factor2,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    saturation_mod = tf.add(saturation,saturation_factor3)
    saturation = clip_ops.clip_by_value(saturation_mod, 0.0, 1.0)

    hsv_altered = array_ops.concat([hue, saturation, value], 3)
    rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

    # brightness
    brightness_factor = random_ops.random_uniform([batch_size],-.25,.25)
    brightness_factor2 = tf.multiply(filter_mask,brightness_factor)
    brightness_factor3 = tf.expand_dims(tf.transpose(tf.tile(tf.reshape(brightness_factor2,[1,1,batch_size]),(crop_size[0],crop_size[1],1)),(2,0,1)),3)
    adjusted = math_ops.add(rgb_altered,math_ops.cast(brightness_factor3,dtypes.float32))

    filtered_batch = clip_ops.clip_by_value(adjusted,0.0,255.0)

    # insert people masks
    num_people_masks = int(batch_size*percent_people)
    mask_inds = np.random.choice(np.arange(0,batch_size),num_people_masks,replace=False)

    start_masks = np.zeros([batch_size, crop_size[0], crop_size[0], 1],dtype='float32')
    start_masks[mask_inds,:,:,:] = 1

    inv_start_masks = np.ones([batch_size, crop_size[0], crop_size[0], 1],dtype='float32')
    inv_start_masks[mask_inds,:,:,:] = 0

    masked_masks = tf.add(inv_start_masks,tf.cast(tf.multiply(people_mask_batch,start_masks),dtype=tf.float32))
    masked_masks2 = tf.cast(tf.tile(masked_masks,[1, 1, 1, 3]),dtype=tf.float32)
    masked_batch = tf.multiply(masked_masks,filtered_batch)

    noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
    final_batch = tf.add(masked_batch,noise)

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

    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    #D = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    D = 1 - tf.reduce_sum(tf.multiply(expanded_a, expanded_b), 2)

    # if not train_data.isOverfitting:
    #     D_max = tf.reduce_max(D)
    #     D_mean, D_var = tf.nn.moments(D, axes=[0,1])
    #     lowest_nonzero_distance = tf.reduce_max(-D)
    #     bottom_thresh = 1.2*lowest_nonzero_distance
    #     top_thresh = (D_max + D_mean)/2.0
    #     bool_mask = tf.logical_and(D>=bottom_thresh,D<=top_thresh)
    #     D = tf.multiply(D,tf.cast(bool_mask,tf.float32))

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
    chain_based_margin[:int(float(batch_size)*fraction_same_chain),:int(float(batch_size)*fraction_same_chain),:] = same_chain_margin

    posDists = tf.reshape(tf.gather_nd(D,posPairInds),(batch_size,num_pos_examples))

    shiftPosDists = tf.reshape(posDists,(1,batch_size,num_pos_examples))
    posDistsRep = tf.tile(shiftPosDists,(batch_size,1,1))

    allDists = tf.tile(tf.expand_dims(D,2),(1,1,num_pos_examples))

    all_loss = tf.maximum(0.,tf.multiply(same_class_mask,posDistsRep - allDists + chain_based_margin))
    non_zero_mask = tf.greater(all_loss, 0)
    non_zero_array = tf.boolean_mask(all_loss, non_zero_mask)
    loss = tf.reduce_mean(all_loss)

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
        batch, labels, chains, ims = train_data.getBatch()
        people_masks = train_data.getPeopleMasks()
        batch_time = time.time() - start_time
        start_time = time.time()
        _, nza, loss_val = sess.run([train_op,non_zero_array, loss], feed_dict={image_batch: batch, people_mask_batch:people_masks})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.6f | non-zero triplets: %d -- (batch creation: %.3f | training: %.3f sec)' % (step, loss_val, nza.shape[0], batch_time,duration)
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
    if len(args) < 11:
        print 'Expected input parameters: fraction_same_chain, same_chain_margin, diff_chain_margin, batch_size,output_size,learning_rate,whichGPU,is_finetuning'
    fraction_same_chain = args[1]
    same_chain_margin = args[2]
    diff_chain_margin = args[3]
    batch_size = args[4]
    output_size = args[5]
    learning_rate = args[6]
    whichGPU = args[7]
    is_finetuning = args[8]
    is_overfitting = args[9]
    pretrained_net = args[10]
    main(fraction_same_chain,same_chain_margin,diff_chain_margin,batch_size,output_size,learning_rate,whichGPU,is_finetuning,is_overfitting,pretrained_net)
