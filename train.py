#!/usr/bin/env python3
#1;95;0c -*- coding: utf-8 -*-
# File: train.py

import glob
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf
import os
import scipy.misc
#---------------------------#
import keras
import keras.backend as K
#---------------------------#
from PIL import Image
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu

from coco import COCODetection
from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model import (
    clip_boxes, decode_bbox_target, encode_bbox_target, crop_and_resize,
    rpn_head, rpn_losses,
    generate_rpn_proposals, sample_fast_rcnn_targets, roi_align,
    fastrcnn_head, fastrcnn_losses, fastrcnn_predictions,
    maskrcnn_head, maskrcnn_loss, secondclassification_head, secondclassification_losses)
from data import (
    get_train_dataflow_coco, get_train_dataflow_davis, get_train_dataflow_mapillary, get_train_dataflow_coco_and_mapillary, get_eval_dataflow,
    get_all_anchors, inverse_get_train_dataflow_davis)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image, print_evaluation_scores, DetectionResult, SecondDetectionResult)
import config
from forward_proto import forward_protobuf

#-------------------------------------------------#
global num_images_video

# parameters that need to be tuned
global alpha_var_l2
global alpha_var_l1
global alpha_l2_norm
global num_epochs
global opt_lr
global alpha_bn

def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu


def get_model_output_names():
    if config.USE_SECOND_HEAD:
        ret = ['final_boxes', 'final_probs', 'final_labels', 'final_posterior', 'second_final_labels',
               'second_final_posterior']
    else:
        ret = ['final_boxes', 'final_probs', 'final_labels', 'final_posterior']
    if config.MODE_MASK:
        ret.append('final_masks')
    if config.EXTRACT_FEATURES:
        ret.append('feature_fastrcnn_pooled')
    return ret
                                
# definition of the model
class Model(ModelDesc):
    def _get_inputs(self):
        ret = [
            InputDesc(tf.float32, (480, 854, 3), 'image'),
            InputDesc(tf.int32, (None, None, config.NUM_ANCHOR), 'anchor_labels'), 
            InputDesc(tf.float32, (None, None, config.NUM_ANCHOR, 4), 'anchor_boxes'), 
            InputDesc(tf.float32, (None, 4), 'gt_boxes'),
            InputDesc(tf.int64, (None,), 'gt_labels')] # all > 0
        if config.USE_SECOND_HEAD:
            ret.append(
                InputDesc(tf.int64, (None,), 'second_gt_labels')
            )
        if config.MODE_MASK and config.INVERSION:
            ret.append(
                InputDesc(tf.uint8, (None, None, None), 'gt_masks') ) # NR_GT x height x width
        if config.PROVIDE_BOXES_AS_INPUT:
            ret.append(
                InputDesc(tf.float32, (None, 4), 'input_boxes')
            )
        return ret

    def _preprocess(self, image):
        image = tf.expand_dims(image, 0)
        #image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def _get_anchors(self, image):
        """
        Returns:
            FSxFSxNAx4 anchors,
        """
        # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
        with tf.name_scope('anchors'):
            all_anchors = tf.constant(get_all_anchors(), name='all_anchors', dtype=tf.float32)
            fm_anchors = tf.slice(
                all_anchors, [0, 0, 0, 0], tf.stack([
                    tf.shape(image)[0] // config.ANCHOR_STRIDE,
                    tf.shape(image)[1] // config.ANCHOR_STRIDE,
                    -1, -1]), name='fm_anchors')
            return fm_anchors

    def _build_graph(self, inputs):
        global alpha_var_l2
        global alpha_var_l1
        global alpha_l2_norm
        global alpha_bn

        is_training = True 

        if config.USE_SECOND_HEAD:
            if config.MODE_MASK and config.INVERSION:
                orig_img, anchor_labels, anchor_boxes, gt_boxes, gt_labels, second_gt_labels, gt_masks = inputs
            else:
                image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, second_gt_labels = inputs

        else:
            if config.MODE_MASK:
                if config.PROVIDE_BOXES_AS_INPUT:
                    image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks, input_boxes = inputs
                else:
                    image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks = inputs
            else:
                image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
    
            second_gt_labels = None
        

        # initialize the tensors as random noise with same shape as input tensor - cutting off the inputs---------------# 
        tf.set_random_seed(0) # for reproducibility
        noise_img = tf.random_normal(tf.shape(orig_img), seed=0)
        image = tf.Variable(initial_value=noise_img, trainable=True, name='img') # is a normal gaussian mean 0, variance 1 
        fm_anchors = self._get_anchors(image)
        image_shape2d_before_resize = tf.shape(image)[:2] 
        image = self._preprocess(image) # 1CHW 
        image_jit = image # define a flipped/jittered image

        # changing of input for forward propagation
        # jittering
        off1 = random.randint(-30, 30)
        off2 = random.randint(-30, 30)
        image_jit = tf.manip.roll(image_jit, shift=[off1, off2], axis=[2, 3])
        # horizontal flipping
        flip = random.random() > 0.5
        if flip:
            image_jit = tf.reverse(image_jit, axis=[3])

        # l2 norm loss
        l2_loss = l2_norm_loss(image_jit)
        l2_loss = tf.identity(l2_loss, name="norm_l2_loss")
        #add_moving_summary(l2_loss)

        # total variance -----------------------------------------------------------------------------------------------#
        var_l1_loss, var_l2_loss = total_var_loss(image_jit) var_l1_loss = tf.identity(var_l1_loss, name="l1_var_loss") 
        #add_moving_summary(var_l1_loss)
        var_l2_loss = tf.identity(var_l2_loss, name="l2_var_loss")
        add_moving_summary(var_l2_loss) 
        #---------------------------------------------------------------------------------------------------------------#

        image_shape2d = tf.shape(image)[2:]
        
        anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)
        featuremap = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])

        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, 1024, config.NUM_ANCHOR)
        decoded_boxes = decode_bbox_target(rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(decoded_boxes, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d)

        if config.PROVIDE_BOXES_AS_INPUT:
            old_height, old_width = image_shape2d_before_resize[0], image_shape2d_before_resize[1]
            new_height, new_width = image_shape2d[0], image_shape2d[1]
            height_scale = new_height / old_height
            width_scale = new_width / old_width
            # TODO: check the order of dimensions!
            scale = tf.stack([width_scale, height_scale, width_scale, height_scale], axis=0)
            proposal_boxes = input_boxes * tf.cast(scale, tf.float32)

        secondclassification_labels = None

        if is_training: 
            # sample proposal boxes in training
            rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt, bg_inds = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)

            if config.USE_SECOND_HEAD:
                secondclassification_labels = tf.stop_gradient(tf.concat(
                    [tf.gather(second_gt_labels, fg_inds_wrt_gt),
                     tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name='second_sampled_labels'))
            boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / config.ANCHOR_STRIDE)
        else:
            # use all proposal boxes in inference
            boxes_on_featuremap = proposal_boxes * (1.0 / config.ANCHOR_STRIDE)
            rcnn_labels = rcnn_sampled_boxes = fg_inds_wrt_gt = None

        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
        def ff_true():
            print("ff true")
            feature_fastrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])    # nxcx7x7
            if config.TRAIN_HEADS_ONLY:
                feature_fastrcnn = tf.stop_gradient(feature_fastrcnn)
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head('fastrcnn', feature_fastrcnn, config.NUM_CLASS)
            return feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits

        def ff_false():
            print("ff false")
            ncls = config.NUM_CLASS
            return tf.zeros([0, 2048, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

        feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
            tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)
        feature_fastrcnn_pooled = tf.reduce_mean(feature_fastrcnn, axis=[2, 3], name="feature_fastrcnn_pooled")

        if config.USE_SECOND_HEAD:
            def if_true():
                secondclassification_label_logits = secondclassification_head('secondclassification', feature_fastrcnn,
                                                                              config.SECOND_NUM_CLASS)
                return secondclassification_label_logits

            def if_false():
                ncls = config.SECOND_NUM_CLASS
                return tf.zeros([0, ncls])
            secondclassification_label_logits = tf.cond(tf.size(boxes_on_featuremap) > 0, if_true, if_false)
        else:
            secondclassification_label_logits = None

        if is_training:
            # rpn loss
            print("Training")
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)

            with tf.name_scope('fg_sample_patch_viz'):
                fg_sampled_patches = crop_and_resize(
                    image, fg_sampled_boxes,
                    tf.zeros_like(fg_inds_wrt_sample, dtype=tf.int32), 300)
                fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
                fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
                #tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
            encoded_boxes = encode_bbox_target(
                matched_gt_boxes,
                fg_sampled_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
            fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
                rcnn_labels, fastrcnn_label_logits,
                encoded_boxes,
                tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

            if config.USE_SECOND_HEAD:
                secondclassification_label_loss = secondclassification_losses(
                    secondclassification_labels, secondclassification_label_logits)
            else:
                secondclassification_label_loss = None

            # mask rcnn loss that helps image take its shape ----------------------------------------------#
            if config.MODE_MASK:
                # maskrcnn loss
                fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
                fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
                mask_logits = maskrcnn_head('maskrcnn', fg_feature, config.NUM_CLASS)   # #fg x #cat x 14x14

                gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
                
                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks_for_fg, 1),
                    fg_sampled_boxes,
                    tf.range(tf.size(fg_inds_wrt_gt)), 14)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
                
            else:
                mrcnn_loss = 0.0  
              
            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn|maskrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            # batch norm extraction -----------------------------------------------------------------------#
            # /home/hs12/.local/lib/python3.6/site-packages/tensorflow/python/ops
            
            # extracting moving means/variances
            variables = tf.global_variables() 
            vars_moving_mean = []
            vars_moving_variance = []
            vars_moving_beta = [] # offset 
            vars_moving_gamma = [] # scale

            for var in variables:
                if ("bn/mean" in var.name):
                    vars_moving_mean.append(var)
                elif ("bn/variance" in var.name):
                    vars_moving_variance.append(var)
                elif ("bn/beta" in var.name):
                    vars_moving_beta.append(var)
                elif ("bn/gamma" in var.name):
                    vars_moving_gamma.append(var)

            # extract estimated means/variances
            all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
            bn_inputs = []
            for tensor in all_tensors:
                if ("Conv2D:0" in tensor.name and "conv" in tensor.name and not "maskrcnn" in tensor.name and not "rpn" in tensor.name):
                    bn_inputs.append(tensor)
            
            num_layers_to_evaluate = len(bn_inputs)

            feature_loss = 0
            for layer_index in range(num_layers_to_evaluate):
                res_layer = bn_inputs[layer_index]               
                avg, var = tf.nn.moments(res_layer, axes=[0, 2, 3]) # input - NCHW

                # define feature loss
                feature_loss += tf.norm(avg - vars_moving_mean[layer_index])
                feature_loss += tf.norm(var - vars_moving_variance[layer_index])
            
            feature_loss = tf.identity(feature_loss, name="bn_feature_loss")
            add_moving_summary(feature_loss)
            
            # define scaled loss terms -------------------------------------------------------------------#
         
            weighted_var_l2_loss = alpha_var_l2 * var_l2_loss
            weighted_var_l1_loss = alpha_var_l1 * var_l1_loss
            weighted_l2_norm_loss = alpha_l2_norm * l2_loss
            weighted_mrcnn_loss = mrcnn_loss
            weighted_feature_loss = alpha_bn * feature_loss
            weighted_var_l2_loss = tf.identity(weighted_var_l2_loss, name="scaled_l2_var_loss")
            weighted_var_l1_loss = tf.identity(weighted_var_l1_loss, name="scaled_l1_var_loss")
            weighted_l2_norm_loss = tf.identity(weighted_l2_norm_loss, name="scaled_norm_l2_loss")
            weighted_mrcnn_loss = tf.identity(weighted_mrcnn_loss, name="scaled_mrcnn_loss")
            weighted_feature_loss = tf.identity(weighted_feature_loss, name="scaled_bn_feature_loss")
            add_moving_summary(weighted_var_l2_loss)
            #add_moving_summary(weighted_var_l1_loss)
            add_moving_summary(weighted_l2_norm_loss)
            add_moving_summary(weighted_mrcnn_loss)
            add_moving_summary(weighted_feature_loss)


            # cost definition ----------------------------------------------------------------------------#
            if config.INVERSION:
                    self.cost = tf.add_n([
                        weighted_feature_loss, weighted_var_l2_loss, weighted_var_l1_loss,
                        weighted_l2_norm_loss, weighted_mrcnn_loss],  'total_cost')
            
            elif config.TRAIN_HEADS_ONLY:
                # don't include the rpn loss
                if config.USE_SECOND_HEAD:
                    self.cost = tf.add_n([
                        fastrcnn_label_loss, fastrcnn_box_loss,
                        secondclassification_label_loss,
                        #mrcnn_loss,
                        wd_cost], 'total_cost')
                else:
                    self.cost = tf.add_n([
                        fastrcnn_label_loss, fastrcnn_box_loss,
                        #mrcnn_loss,
                        wd_cost], 'total_cost')
            else:
                if config.USE_SECOND_HEAD:
                    self.cost = tf.add_n([
                        rpn_label_loss, rpn_box_loss,
                        fastrcnn_label_loss, fastrcnn_box_loss,
                        secondclassification_label_loss,
                        #mrcnn_loss,
                        wd_cost], 'total_cost')
                else:
                    self.cost = tf.add_n([
                        rpn_label_loss, rpn_box_loss,
                        fastrcnn_label_loss, fastrcnn_box_loss,
                        #mrcnn_loss,
                        wd_cost], 'total_cost')

            add_moving_summary(self.cost)
            
            # only want input image to be trainable, remove other variables from being trained-------------------------------#
            trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
            length = len(trainable_collection)

            i = 1
            while i < length:
                trainable_collection.pop(length - 1)
                length += -1

            # viewing optimized image
            mean = [0.485, 0.456, 0.406] # rgb std = [0.229, 0.224, 0.225]
            image_view = image
            image_view = clipping(mean, std, image_view)
            image_view = denormalize(mean, std, image_view)
            image_view = tf.transpose(image_view, [0, 2, 3, 1])
            image_view = tf.reverse(image_view, axis=[-1]) # reverse bgr to rgb tf.summary.image('optimized_images', image_view, max_outputs=82)
                         
        else:
            #print("not training")
            label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
            
            anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, config.NUM_CLASS - 1, 1])   # #proposal x #Cat x 4
            decoded_boxes = decode_bbox_target(
                fastrcnn_box_logits /
                tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            # indices: Nx2. Each index into (#proposal, #category)
            pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
            final_probs = tf.identity(final_probs, 'final_probs')
            final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
            final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
            pred_indices_only_boxes = pred_indices[:, 1]
            final_posterior = tf.gather(label_probs, pred_indices_only_boxes, name='final_posterior')

            if config.USE_SECOND_HEAD:
                second_label_probs = tf.nn.softmax(secondclassification_label_logits,
                                                   name='secondclassification_all_probs')  # #proposal x #Class
                second_final_posterior = tf.gather(second_label_probs, pred_indices_only_boxes,
                                                   name='second_final_posterior')
                second_final_labels = tf.add(tf.argmax(final_posterior, -1), 1, name='second_final_labels')

            if config.MODE_MASK:
                # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
                def f1():
                    roi_resized = roi_align(featuremap, final_boxes * (1.0 / config.ANCHOR_STRIDE), 14)
                    feature_maskrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])
                    mask_logits = maskrcnn_head(
                        'maskrcnn', feature_maskrcnn, config.NUM_CLASS)   # #result x #cat x 14x14
                    indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                    return tf.sigmoid(final_mask_logits)

                final_masks = tf.cond(tf.size(final_probs) > 0, f1, lambda: tf.zeros([0, 14, 14]))
                tf.identity(final_masks, name='final_masks')

    def _get_optimizer(self):    
        global opt_lr
        # optimizer defined to optimize inputs
        lr = tf.get_variable('learning_rate', initializer=opt_lr, trainable=False) 
        opt = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-08) 
        '''
        with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
            lr = tf.get_variable('learning_rate', initializer=0.001, trainable=True) 
            tf.summary.scalar('learning_rate', lr)

            factor = get_batch_factor()
            if factor != 1:
                lr = lr / float(factor)
                opt = tf.train.MomentumOptimizer(lr, 0.9)
                opt = optimizer.AccumGradOptimizer(opt, factor)
            else:
                opt = tf.train.MomentumOptimizer(lr, 0.9)
        '''
       # tf.get_variable_scope().reuse_variables()
        return opt


# clipping of images
def clipping(mean, std, image):
    # clip dimension 0
    m0, s0 = mean[2], std[2]
    clip0 = tf.clip_by_value(image[:,0], -m0/s0, (1-m0)/s0) clip0 = tf.expand_dims(clip0, 1)

    # clip dimension 1
    m1, s1 = mean[1], std[1]
    clip1 = tf.clip_by_value(image[:,1], -m1/s1, (1-m1)/s1)
    clip1 = tf.expand_dims(clip1, 1)

    # clip dimension 2
    m2, s2 = mean[0], std[0]
    clip2 = tf.clip_by_value(image[:,2], -m2/s2, (1-m2)/s2)
    clip2 = tf.expand_dims(clip2, 1)

    image = tf.concat([clip0, clip1, clip2], 1)
    
    return image

# denormalize images
def denormalize(mean, std, image):
# denormalize dimension 0
    m0, s0 = mean[2], std[2]
    clip0 = tf.clip_by_value(image[:,0] * s0 + m0, 0, 1) clip0 = tf.expand_dims(clip0, 1)

    # clip dimension 1
    m1, s1 = mean[1], std[1]
    clip1 = tf.clip_by_value(image[:,1] * s1 + m1, 0, 1)
    clip1 = tf.expand_dims(clip1, 1)

    # clip dimension 2
    m2, s2 = mean[0], std[0]
    clip2 = tf.clip_by_value(image[:,2] * s2 + m2, 0, 1)
    clip2 = tf.expand_dims(clip2, 1)
    image = tf.concat([clip0, clip1, clip2], 1)
    image = tf.scalar_mul(255.0, image)

    return image


# calculating the variance image prior loss
def total_var_loss(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    var_l2_loss = tf.norm(diff1) + tf.norm(diff2) + tf.norm(diff3) + tf.norm(diff4)
    var_l1_loss = tf.reduce_mean(tf.abs(diff1)) + tf.reduce_mean(tf.abs(diff2)) + \
                  tf.reduce_mean(tf.abs(diff3)) + tf.reduce_mean(tf.abs(diff4))
    
    return var_l1_loss, var_l2_loss


# finding l2 norm loss
def l2_norm_loss(inputs_jit):
    tensor = tf.reshape(inputs_jit, [1, -1])
    l2_loss = tf.norm(tensor, axis=1)
    l2_loss = tf.reduce_mean(l2_loss)
    return l2_loss
    

def visualize(model_path, nr_visualize=50, output_dir='output'):
    df = get_train_dataflow_coco()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_rpn_proposals/boxes',
            'generate_rpn_proposals/probs',
            'fastrcnn_all_probs',
            'final_boxes',
            'final_probs',
            'final_labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df.get_data()), nr_visualize):
            img, _, _, gt_boxes, gt_labels = dp

            rpn_boxes, rpn_scores, all_probs, \
                final_boxes, final_probs, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_probs[good_proposals_ind])

            if config.USE_SECOND_HEAD:
                results = [SecondDetectionResult(*args) for args in
                           zip(final_boxes, final_probs, final_labels,
                               [None] * len(final_labels))]
            else:
                results = [DetectionResult(*args) for args in
                           zip(final_boxes, final_probs, final_labels,
                               [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_on_dataflow(
        df, lambda img: detect_one_image(img, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    print_evaluation_scores(output_file)


def convert_results_to_json(results, img_idx):
    img_res = []
    for r in results:
        box = r.box
        from coco import COCOMeta
        # cat_id = COCOMeta.class_id_to_category_id[r.class_id]
        if config.USE_SECOND_HEAD:
            cc = r.second_class_id
            # second_cat_id = COCOMeta.second_class_id_to_category_id[cc]
        box[2] -= box[0]
        box[3] -= box[1]
        if config.USE_SECOND_HEAD:
            res = {
                # 'image_id': img_idx,
                # 'category_id': cat_id,
                'bbox': list(map(lambda x: float(round(x, 1)), box)),
                'score': float(round(r.score, 2)),
                # 'posterior': r.posterior.tolist(),
                # 'second_category_id': second_cat_id,
                # 'second_posterior': r.second_posterior.tolist()
            }
        else:
            res = {
                # 'image_id': img_idx,
                # 'category_id': cat_id,
                'bbox': list(map(lambda x: float(round(x, 1)), box)),
                'score': float(round(r.score, 2)),
                # 'posterior': r.posterior.tolist(),
            }
        if config.EXTRACT_FEATURES:
            # TODO: we need to write it out in a different way! this is too large (33MB for 1 image with 1000 proposals)
            res["features"] = [float(x) for x in r.feature_fastrcnn_pooled]
        # also append segmentation to results
        if r.mask is not None:
            import pycocotools.mask as cocomask
            rle = cocomask.encode(
                np.array(r.mask[:, :, None], order='F'))[0]
            rle['counts'] = rle['counts'].decode('ascii')
            res['segmentation'] = rle
        img_res.append(res)
    return img_res


def forward(pred_func, output_folder, forward_dataset, generic_images_folder=None, generic_images_pattern=None):
    if forward_dataset.lower() == "davis":
        # name = "/home/luiten/vision/PReMVOS/home_data/%s/images/*"%config.DAVIS_NAME
        # imgs = glob.glob(name)
        # seq_idx = -2

        # name = config.DAVIS_NAME
        # imgs = glob.glob(name+"/*/*")
        # seq_idx = -2

        name = config.DAVIS_NAME
        pre = '/'.join(name.split('/')[:-1])
        print(name)
        imgs = []
        f = open(name, 'r')
        while True:
            x = f.readline()
            x = x.rstrip()
            if not x: break
            print(pre+'/'+x)
            imgs = imgs + glob.glob(pre+'/'+x + '/*')

        # for file in open(name):
        #     imgs = imgs + glob.glob(file+'/*')
        print(imgs)
        seq_idx = -2

    elif forward_dataset.lower() == "oxford":
        imgs = ["/fastwork/" + os.environ["USER"] + "/mywork/" + x.strip() for x in
                open("/home/" + os.environ["USER"] +
                     "/vision/TrackingAnnotationTool/to_annotate_single_imgs.txt").readlines() if len(x.strip()) > 0]
        seq_idx = -4
    elif forward_dataset.lower() == "kitti_tracking":
        imgs = glob.glob("/fastwork/" + os.environ["USER"] +
                         "/mywork/data/kitti_training_minimum/training/image_02/*/*.png")
        seq_idx = -2
    elif forward_dataset.lower() == "generic":
        generic_images_pattern = generic_images_pattern.replace("//", "/")
        assert generic_images_folder is not None, "For a generic dataset, a data folder must be given."
        assert generic_images_pattern is not None, "For a generic dataset, an image pattern must be given."
        # Find the images in the folder
        imgs = sorted(glob.glob(os.path.join(generic_images_folder, generic_images_pattern)))
        # The sequence (if there is any) is assumed to be the folder the images are in
        #seq_idx = -1 - generic_images_pattern.count("/")
        seq_idx = None
    elif forward_dataset.lower() == "protobuf":
        forward_protobuf(pred_func, output_folder, forward_dataset, generic_images_folder, generic_images_pattern)
    else:
        assert False, ("Unknown dataset", forward_dataset)
    tf.gfile.MakeDirs(output_folder)
    n_total = len(imgs)
    for idx, img in enumerate(imgs):
        print(idx, "/", n_total)
        if forward_dataset.lower() == "generic":
            seq = "/".join(img.replace(generic_images_folder, "").split("/")[:-1])
            seq_folder = output_folder + "/" + seq
        else:
            seq = img.split("/")[seq_idx]
            seq_folder = output_folder + "/" + seq
        img_filename = img.split("/")[-1]
        tf.gfile.MakeDirs(seq_folder)

        output_filename = seq_folder + "/" + img_filename.replace(".jpg", ".json").replace(".png", ".json")
        if os.path.isfile(output_filename):
            print("skipping", output_filename, "because it already exists")
            continue
        else:
            print(output_filename)

        img_val = cv2.imread(img, cv2.IMREAD_COLOR)
        assert img_val is not None, ("unable to load", img)
        if config.PROVIDE_BOXES_AS_INPUT:
            # TODO get some reasonable boxes (not sure which format)
            input_boxes = np.array([[202.468384, 566.15271, 999, 800], [0, 0, 100, 100], [100, 100, 200, 200]],
                                   np.float32)
            results = detect_one_image(img_val, pred_func, input_boxes)
        #This is stage 4! (the else block)
        else:
            results = detect_one_image(img_val, pred_func)
        print(len(results))

        # store visualization (slow + optional)
        if args.forward_visualization:
            final = draw_final_outputs(img_val, results)
            viz = np.concatenate((img_val, final), axis=1)
            #tpviz.interactive_imshow(viz)
            viz_output_filename = seq_folder + "/" + img_filename #.png
            cv2.imwrite(viz_output_filename, viz)

        # store as json
        results_json = convert_results_to_json(results, idx)
        with open(output_filename, 'w') as f:
            json.dump(results_json, f)


def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    print(len(results))
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)


class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'],
            get_model_output_names())
        self.df = get_eval_dataflow()

    def _before_train(self):
        print("BEFORE TRAIN")
        EVAL_TIMES = 5  # eval 5 times during training
        interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1, EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        all_results = eval_on_dataflow(self.df, lambda img: detect_one_image(img, self.pred))
        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        print_evaluation_scores(output_file)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')

    parser.add_argument('--agnostic', action='store_true', help='use class-agnostic model')
    parser.add_argument('--heads_only', action='store_true')
    parser.add_argument('--second_head', action='store_true')
    parser.add_argument('--mapillary', action='store_true')
    parser.add_argument('--davis', action='store_true')
    parser.add_argument('--coco_and_mapillary', action='store_true')

    parser.add_argument('--forward', help='path to the output json eval file')
    parser.add_argument('--forward_dataset', default="DAVIS",
                        help='DAVIS|Oxford|generic, The dataset chosen for computation. Use "generic" and supply a '
                             'generic_images_folder argument for computing masks for images that do not belong to any '
                             'predefined dataset.')
    parser.add_argument('--generic_images_folder', help='folder to search for input images for a generic dataset')
    parser.add_argument('--generic_images_pattern', default="*.png")
    parser.add_argument('--forward_visualization', action='store_true')
    parser.add_argument('--original_lr_schedule', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--extract_features', action='store_true')
    parser.add_argument('--provide_boxes_as_input', action='store_true')
    parser.add_argument('--davis_name', help='the davis sequence to use')

    # add the parameters for tuning
    parser.add_argument('--num_epochs', type=float, help = 'number of epochs')
    parser.add_argument('--learning_rate_opt', type=float, help = 'learning rate')
    parser.add_argument('--scaling_tv', type=float, help = 'scaling total variance')
    parser.add_argument('--scaling_bn', type=float, help = 'scaling bn feature')

    args = parser.parse_args()
    
    args.davis = True # added by Hari for DAVIS set
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.agnostic:
        config.CATEGORY_AGNOSTIC = True
        config.NUM_CLASS = 2

    if args.heads_only:
        config.TRAIN_HEADS_ONLY = True

    if args.second_head:
        config.USE_SECOND_HEAD = True

    if args.extract_features:
        config.EXTRACT_FEATURES = True

    if args.provide_boxes_as_input:
        config.PROVIDE_BOXES_AS_INPUT = True

    if args.mapillary:
        assert not args.coco_and_mapillary
        config.USE_MAPILLARY = True
        config.SECOND_NUM_CLASS = len(config.MAPILLARY_CAT_IDS_TO_USE) + 1

    if args.davis:
        assert not args.coco_and_mapillary and not args.mapillary
        config.USE_DAVIS = True
        config.SECOND_NUM_CLASS = 81

    config.DAVIS_NAME = args.davis_name

    if args.coco_and_mapillary:
        assert config.USE_SECOND_HEAD, "so far only works with second head due to void label handling"
        config.USE_COCO_AND_MAPILLARY = True
        config.SECOND_NUM_CLASS = 81

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.predict:
        config.RESULT_SCORE_THRESH = 0.01
        config.FASTRCNN_NMS_THRESH = 0.5
        config.RESULTS_PER_IM = 500

    if args.forward:
        config.MODE_MASK = False
    else:
        config.MODE_MASK = True

    config.MODE_MASK = True
    config.INVERSION = True ######
    if not (args.visualize or args.evaluate or args.predict or args.forward):

        # added by Hari to debug and find path 
        print("Stage 1!\n")
        print("Args.visualize " + str(args.visualize))
        print("Args.evaluate " + str(args.evaluate))
        print("Args.predict " + str(args.predict))
        print("Args.forward " + str(args.forward))
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()
        if args.visualize:
            visualize(args.load)
        else:
            print("Stage 2!\n")
            input_names = ['image']
            if config.PROVIDE_BOXES_AS_INPUT:
                input_names.append('input_boxes')
            pred = OfflinePredictor(PredictConfig(
                model=Model(),
                session_init=get_model_loader(args.load),
                input_names=input_names,
                output_names=get_model_output_names()))

            if args.evaluate:
                assert args.evaluate.endswith('.json')
                offline_evaluate(pred, args.evaluate)
            elif args.forward:
                print("Stage 3!\n")
                assert not args.forward.endswith('.json')
                # TODO: maybe we can get rid of this dependency to COCO
                COCODetection(config.BASEDIR, 'minival2014')  # to load the class names into caches
                forward(pred, args.forward, args.forward_dataset, args.generic_images_folder,
                        args.generic_images_pattern)
            elif args.predict:
                COCODetection(config.BASEDIR, 'train2014')   # to load the class names into caches
                predict(pred, args.predict)
                
    else:

        # setting parameter values------------------------------------------------------------#
        alpha_tv = args.scaling_tv
        print("Alpha total variance: " + str(alpha_tv))
        num_epochs = args.num_epochs
        print("Number of epochs: " + str(num_epochs))
        opt_lr = args.learning_rate_opt
        print("Learning rate: " + str(opt_lr))
        alpha_bn = args.scaling_bn
        print("Alpha bn feature: " + str(alpha_bn))

        folder_name = 'atv' + str(alpha_tv) + '_epochs' + str(num_epochs) + '_lr' + str(opt_lr) + '_bn' + str(alpha_bn)
        #-------------------------------------------------------------------------------------#
        delete = True

        # logging of remainder of training
        args.logdir = '/tigress/hs12/Thesis/train_log'
        print(args.logdir)
        
        args.logdir = args.logdir + '/' + str(folder_name)
        output_dir = '/home/hs12/Thesis/synthesized_images/bear'

        if os.path.isdir(args.logdir) and delete: # set this to false if you want to save results
            shutil.rmtree(args.logdir)
        
        if os.path.isdir(output_dir + '/' + str(folder_name)) and delete:
            shutil.rmtree(output_dir + '/' + str(folder_name))

        if args.resume:
            logger.set_logger_dir(args.logdir, action="k")
        else:
            logger.set_logger_dir(args.logdir)
        
        print_config()
        
        stepnum = 300 # originally 300
        warmup_epoch = max(math.ceil(500.0 / stepnum), 5)
        factor = get_batch_factor() 

        starting_epoch = 1
        #end_num = 938
        #end_num = 140625
        max_epoch = num_epochs # originally end_num * factor // stepnum
        #max_epoch = end_num * factor // stepnum 

        session_init = get_model_loader(args.load) if args.load else None
        if args.resume and session_init is not None:
            starting_step = int(session_init.path.split("/")[-1].split("-")[-1])
            starting_epoch = starting_step // stepnum
            print("starting at", starting_epoch)
#        if config.CATEGORY_AGNOSTIC and not args.resume:
 #           print("Warning, removing class-specific variables from initialization")
  #          del session_init._prms["fastrcnn/class/W:0"]
   #         del session_init._prms["fastrcnn/class/W:0"]
    #        del session_init._prms["fastrcnn/class/b:0"]
     #       del session_init._prms["fastrcnn/box/W:0"]
      #      del session_init._prms["fastrcnn/box/b:0"]
       #     del session_init._prms["maskrcnn/conv/W:0"]
        #    del session_init._prms["maskrcnn/conv/b:0"]

        if args.original_lr_schedule:
            print("using original learning rate schedule")
            lr_callbacks = [# linear warmup
                            ScheduledHyperParamSetter(
                                'learning_rate',
                                [(0, 3e-3), (warmup_epoch * factor, 1e-2)], interp='linear'),
                            # step decay
                            ScheduledHyperParamSetter(
                                'learning_rate',
                                [(warmup_epoch * factor, 1e-2),
                                 (150000 * factor // stepnum, 1e-3),
                                 (230000 * factor // stepnum, 1e-4)])]
        else:
            print("Using constant learning rate of " + str(opt_lr))
            #lr_callbacks = [ScheduledHyperParamSetter('learning_rate', [(0, 1e-3)])]
            lr_callbacks = [ScheduledHyperParamSetter('learning_rate', [(0, opt_lr)])]

        if config.USE_COCO_AND_MAPILLARY:
            train_dataflow = get_train_dataflow_coco_and_mapillary(add_mask=config.MODE_MASK)
            eval_callbacks = []
        elif config.USE_MAPILLARY:
            train_dataflow = get_train_dataflow_mapillary(add_mask=config.MODE_MASK)
            eval_callbacks = []
        elif config.USE_DAVIS and config.INVERSION:
            train_dataflow = inverse_get_train_dataflow_davis(add_mask=config.MODE_MASK)
            num_images_video = train_dataflow.size()
            print("Inversion size of the dataflow: " + str(train_dataflow.size()))
            eval_callbacks = []
        elif config.USE_DAVIS:
            train_dataflow = get_train_dataflow_davis(add_mask=config.MODE_MASK)
            print("Size of the dataflow: " + str(train_dataflow.size()))
            eval_callbacks = []
        else:
            train_dataflow = get_train_dataflow_coco(add_mask=config.MODE_MASK)
            eval_callbacks = [EvalCallback()]
        
        # overwriting stepnum
        #stepnum = num_images_video
        stepnum = 16

        cfg = TrainConfig(
            model=Model(),
            data=QueueInput(train_dataflow),
            callbacks=[
                [ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1)]
                + lr_callbacks +
                eval_callbacks +
                [GPUUtilizationTracker()]
            ],
             steps_per_epoch=stepnum,
             starting_epoch=starting_epoch,
             max_epoch=max_epoch,
             session_init=session_init,
           # steps_per_epoch=stepnum,
           # starting_epoch=starting_epoch,
           # max_epoch=280000 * factor // stepnum,
           # session_init=session_init,
        )

        print("steps_per_epoch: " + str(stepnum))
        print("starting_epoch: " + str(starting_epoch))
        print("max_epoch: " + str(max_epoch))
       
        print("Number of GPUS: " + str(get_nr_gpu()))
        #trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu())
        trainer = SimpleTrainer()

        # placeholders for saving images - need to be instantiated before creating graph
        image_str = tf.placeholder(tf.string)
        im_tf = tf.image.decode_image(image_str)

        # /home/hs12/.local/lib/python3.6/site-packages/tensorpack/train
        launch_train_with_config(cfg, trainer)
        
        # save final optimized images
        fn = '/tigress/hs12/Thesis/train_log/' + str(folder_name)
        for filename in os.listdir(fn):
            if str(filename[0:6]) == 'events':
                event_file = filename
                fn = fn + "/" + str(event_file)

        #output_dir = '/home/hs12/Thesis/synthesized_images/bear' 
        os.mkdir(output_dir + '/' + str(folder_name))
        output_dir = output_dir + '/' + str(folder_name)
        sess = tf.InteractiveSession()                                                                                                         
        with sess.as_default():                                                                                                                 
            count = 0
            for e in tf.train.summary_iterator(fn):
                for v in e.summary.value:
                    if "image" in v.tag:

                        '''
                        if count % 10 != 0:
                            count += 1
                            continue
                        '''
                        im = im_tf.eval({image_str: v.image.encoded_image_string})
                        output_fn = os.path.realpath('{}/image_{:02d}.png'.format(output_dir, count))
                        print("Saving '{}'".format(output_fn))
                        scipy.misc.imsave(output_fn, im)
                        count += 1
