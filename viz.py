#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py

from six.moves import zip
import numpy as np
from PIL import Image

from tensorpack.utils import viz
from tensorpack.utils.palette import PALETTE_RGB

from utils.box_ops import get_iou_callable
from utils.np_box_ops import iou as np_iou
import config


def draw_annotation(img, boxes, klass, is_crowd=None):
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = config.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(config.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    #box_iou_float = get_iou_callable()
    #box_ious = bbox_iou_float(gt_boxes, proposals)    # ng x np
    box_ious = np_iou(gt_boxes, proposals)  # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :3]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    tags = list(map(str, proposal_scores[good_proposals_ind]))
    img = viz.draw_boxes(img, proposals, tags)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(config.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    tags = []
    for r in results:
        if config.KAGGLE: #True
            tags.append(
                "{},{:.2f}".format("object", r.score))
        else:
            tags.append(
                "{},{:.2f}".format(config.CLASS_NAMES[r.class_id], r.score))
    #boxes = np.asarray([r.box for r in results])
    #ret = viz.draw_boxes(img, boxes, tags)
        
    im = np.zeros(img.shape) # img is numpy array
    
    for r in results:
        if r.mask is not None: 
            print("Drawing Mask")
            print(r.mask)
            ret = draw_mask(im, r.mask)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[4][::-1]
        #x = np.random.choice(len(PALETTE_RGB))
        #color = PALETTE_RGB[x][::-1]
        #print(x)
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im

# added to visualize pngs
def save_with_pascal_colormap(filename, arr):
  colmap = (np.array(pascal_colormap) * 255).round().astype("uint8")
  palimage = Image.new('P', (16, 16))
  palimage.putpalette(list(colmap))
  im = Image.fromarray(np.squeeze(arr.astype("uint8")))
  im2 = im.quantize(palette=palimage)
  im2.save(filename)


def save_pngs(img, results, path):
    png = np.zeros((480,854))
    
    mask_count = 0
    for r in results:
      if r.mask is not None:
          r_mask_resized = np.resize((r.mask), (480,854))
          png[r_mask_resized.astype("bool")] = mask_count + 1
          mask_count = mask_count + 1

    output_fn = path.replace(".jpg", ".png")
    save_with_pascal_colormap(output_fn, png)

  #output_fol = '/'.join(output_fn.split('/')[:-1])
  #if not os.path.exists(output_fol):
    #os.makedirs(output_fol)
  #save_with_pascal_colormap(output_fn, png)


pascal_colormap = [
    0     ,         0,         0,
    0.5020,         0,         0,
         0,    0.5020,         0,
    0.5020,    0.5020,         0,
         0,         0,    0.5020,
    0.5020,         0,    0.5020,
         0,    0.5020,    0.5020,
    0.5020,    0.5020,    0.5020,
    0.2510,         0,         0,
    0.7529,         0,         0,
    0.2510,    0.5020,         0,
    0.7529,    0.5020,         0,
    0.2510,         0,    0.5020,
    0.7529,         0,    0.5020,
    0.2510,    0.5020,    0.5020,
    0.7529,    0.5020,    0.5020,
         0,    0.2510,         0,
    0.5020,    0.2510,         0,
         0,    0.7529,         0,
    0.5020,    0.7529,         0,
         0,    0.2510,    0.5020,
    0.5020,    0.2510,    0.5020,
         0,    0.7529,    0.5020,
    0.5020,    0.7529,    0.5020,
    0.2510,    0.2510,         0,
    0.7529,    0.2510,         0,
    0.2510,    0.7529,         0,
    0.7529,    0.7529,         0,
    0.2510,    0.2510,    0.5020,
    0.7529,    0.2510,    0.5020,
    0.2510,    0.7529,    0.5020,
    0.7529,    0.7529,    0.5020,
         0,         0,    0.2510,
    0.5020,         0,    0.2510,
         0,    0.5020,    0.2510,
    0.5020,    0.5020,    0.2510,
         0,         0,    0.7529,
    0.5020,         0,    0.7529,
         0,    0.5020,    0.7529,
    0.5020,    0.5020,    0.7529,
    0.2510,         0,    0.2510,
    0.7529,         0,    0.2510,
    0.2510,    0.5020,    0.2510,
    0.7529,    0.5020,    0.2510,
    0.2510,         0,    0.7529,
    0.7529,         0,    0.7529,
    0.2510,    0.5020,    0.7529,
    0.7529,    0.5020,    0.7529,
         0,    0.2510,    0.2510,
    0.5020,    0.2510,    0.2510,
         0,    0.7529,    0.2510,
    0.5020,    0.7529,    0.2510,
         0,    0.2510,    0.7529,
    0.5020,    0.2510,    0.7529,
         0,    0.7529,    0.7529,
    0.5020,    0.7529,    0.7529,
    0.2510,    0.2510,    0.2510,
    0.7529,    0.2510,    0.2510,
    0.2510,    0.7529,    0.2510,
    0.7529,    0.7529,    0.2510,
    0.2510,    0.2510,    0.7529,
    0.7529,    0.2510,    0.7529,
    0.2510,    0.7529,    0.7529,
    0.7529,    0.7529,    0.7529,
    0.1255,         0,         0,
    0.6275,         0,         0,
    0.1255,    0.5020,         0,
    0.6275,    0.5020,         0,
    0.1255,         0,    0.5020,
    0.6275,         0,    0.5020,
    0.1255,    0.5020,    0.5020,
    0.6275,    0.5020,    0.5020,
    0.3765,         0,         0,
    0.8784,         0,         0,
    0.3765,    0.5020,         0,
    0.8784,    0.5020,         0,
    0.3765,         0,    0.5020,
    0.8784,         0,    0.5020,
    0.3765,    0.5020,    0.5020,
    0.8784,    0.5020,    0.5020,
    0.1255,    0.2510,         0,
    0.6275,    0.2510,         0,
    0.1255,    0.7529,         0,
    0.6275,    0.7529,         0,
    0.1255,    0.2510,    0.5020,
    0.6275,    0.2510,    0.5020,
    0.1255,    0.7529,    0.5020,
    0.6275,    0.7529,    0.5020,
    0.3765,    0.2510,         0,
    0.8784,    0.2510,         0,
    0.3765,    0.7529,         0,
    0.8784,    0.7529,         0,
    0.3765,    0.2510,    0.5020,
    0.8784,    0.2510,    0.5020,
    0.3765,    0.7529,    0.5020,
    0.8784,    0.7529,    0.5020,
    0.1255,         0,    0.2510,
    0.6275,         0,    0.2510,
    0.1255,    0.5020,    0.2510,
    0.6275,    0.5020,    0.2510,
    0.1255,         0,    0.7529,
    0.6275,         0,    0.7529,
    0.1255,    0.5020,    0.7529,
    0.6275,    0.5020,    0.7529,
    0.3765,         0,    0.2510,
    0.8784,         0,    0.2510,
    0.3765,    0.5020,    0.2510,
    0.8784,    0.5020,    0.2510,
    0.3765,         0,    0.7529,
    0.8784,         0,    0.7529,
    0.3765,    0.5020,    0.7529,
    0.8784,    0.5020,    0.7529,
    0.1255,    0.2510,    0.2510,
    0.6275,    0.2510,    0.2510,
    0.1255,    0.7529,    0.2510,
    0.6275,    0.7529,    0.2510,
    0.1255,    0.2510,    0.7529,
    0.6275,    0.2510,    0.7529,
    0.1255,    0.7529,    0.7529,
    0.6275,    0.7529,    0.7529,
    0.3765,    0.2510,    0.2510,
    0.8784,    0.2510,    0.2510,
    0.3765,    0.7529,    0.2510,
    0.8784,    0.7529,    0.2510,
    0.3765,    0.2510,    0.7529,
    0.8784,    0.2510,    0.7529,
    0.3765,    0.7529,    0.7529,
    0.8784,    0.7529,    0.7529,
         0,    0.1255,         0,
    0.5020,    0.1255,         0,
         0,    0.6275,         0,
    0.5020,    0.6275,         0,
         0,    0.1255,    0.5020,
    0.5020,    0.1255,    0.5020,
         0,    0.6275,    0.5020,
    0.5020,    0.6275,    0.5020,
    0.2510,    0.1255,         0,
    0.7529,    0.1255,         0,
    0.2510,    0.6275,         0,
    0.7529,    0.6275,         0,
    0.2510,    0.1255,    0.5020,
    0.7529,    0.1255,    0.5020,
    0.2510,    0.6275,    0.5020,
    0.7529,    0.6275,    0.5020,
         0,    0.3765,         0,
    0.5020,    0.3765,         0,
         0,    0.8784,         0,
    0.5020,    0.8784,         0,
         0,    0.3765,    0.5020,
    0.5020,    0.3765,    0.5020,
         0,    0.8784,    0.5020,
    0.5020,    0.8784,    0.5020,
    0.2510,    0.3765,         0,
    0.7529,    0.3765,         0,
    0.2510,    0.8784,         0,
    0.7529,    0.8784,         0,
    0.2510,    0.3765,    0.5020,
    0.7529,    0.3765,    0.5020,
    0.2510,    0.8784,    0.5020,
    0.7529,    0.8784,    0.5020,
         0,    0.1255,    0.2510,
    0.5020,    0.1255,    0.2510,
         0,    0.6275,    0.2510,
    0.5020,    0.6275,    0.2510,
         0,    0.1255,    0.7529,
    0.5020,    0.1255,    0.7529,
         0,    0.6275,    0.7529,
    0.5020,    0.6275,    0.7529,
    0.2510,    0.1255,    0.2510,
    0.7529,    0.1255,    0.2510,
    0.2510,    0.6275,    0.2510,
    0.7529,    0.6275,    0.2510,
    0.2510,    0.1255,    0.7529,
    0.7529,    0.1255,    0.7529,
    0.2510,    0.6275,    0.7529,
    0.7529,    0.6275,    0.7529,
         0,    0.3765,    0.2510,
    0.5020,    0.3765,    0.2510,
         0,    0.8784,    0.2510,
    0.5020,    0.8784,    0.2510,
         0,    0.3765,    0.7529,
    0.5020,    0.3765,    0.7529,
         0,    0.8784,    0.7529,
    0.5020,    0.8784,    0.7529,
    0.2510,    0.3765,    0.2510,
    0.7529,    0.3765,    0.2510,
    0.2510,    0.8784,    0.2510,
    0.7529,    0.8784,    0.2510,
    0.2510,    0.3765,    0.7529,
    0.7529,    0.3765,    0.7529,
    0.2510,    0.8784,    0.7529,
    0.7529,    0.8784,    0.7529,
    0.1255,    0.1255,         0,
    0.6275,    0.1255,         0,
    0.1255,    0.6275,         0,
    0.6275,    0.6275,         0,
    0.1255,    0.1255,    0.5020,
    0.6275,    0.1255,    0.5020,
    0.1255,    0.6275,    0.5020,
    0.6275,    0.6275,    0.5020,
    0.3765,    0.1255,         0,
    0.8784,    0.1255,         0,
    0.3765,    0.6275,         0,
    0.8784,    0.6275,         0,
    0.3765,    0.1255,    0.5020,
    0.8784,    0.1255,    0.5020,
    0.3765,    0.6275,    0.5020,
    0.8784,    0.6275,    0.5020,
    0.1255,    0.3765,         0,
    0.6275,    0.3765,         0,
    0.1255,    0.8784,         0,
    0.6275,    0.8784,         0,
    0.1255,    0.3765,    0.5020,
    0.6275,    0.3765,    0.5020,
    0.1255,    0.8784,    0.5020,
    0.6275,    0.8784,    0.5020,
    0.3765,    0.3765,         0,
    0.8784,    0.3765,         0,
    0.3765,    0.8784,         0,
    0.8784,    0.8784,         0,
    0.3765,    0.3765,    0.5020,
    0.8784,    0.3765,    0.5020,
    0.3765,    0.8784,    0.5020,
    0.8784,    0.8784,    0.5020,
    0.1255,    0.1255,    0.2510,
    0.6275,    0.1255,    0.2510,
    0.1255,    0.6275,    0.2510,
    0.6275,    0.6275,    0.2510,
    0.1255,    0.1255,    0.7529,
    0.6275,    0.1255,    0.7529,
    0.1255,    0.6275,    0.7529,
    0.6275,    0.6275,    0.7529,
    0.3765,    0.1255,    0.2510,
    0.8784,    0.1255,    0.2510,
    0.3765,    0.6275,    0.2510,
    0.8784,    0.6275,    0.2510,
    0.3765,    0.1255,    0.7529,
    0.8784,    0.1255,    0.7529,
    0.3765,    0.6275,    0.7529,
    0.8784,    0.6275,    0.7529,
    0.1255,    0.3765,    0.2510,
    0.6275,    0.3765,    0.2510,
    0.1255,    0.8784,    0.2510,
    0.6275,    0.8784,    0.2510,
    0.1255,    0.3765,    0.7529,
    0.6275,    0.3765,    0.7529,
    0.1255,    0.8784,    0.7529,
    0.6275,    0.8784,    0.7529,
    0.3765,    0.3765,    0.2510,
    0.8784,    0.3765,    0.2510,
    0.3765,    0.8784,    0.2510,
    0.8784,    0.8784,    0.2510,
    0.3765,    0.3765,    0.7529,
    0.8784,    0.3765,    0.7529,
    0.3765,    0.8784,    0.7529,
    0.8784,    0.8784,    0.7529]
