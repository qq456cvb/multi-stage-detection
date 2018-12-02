import numpy as np
from scipy import interpolate
import tensorflow as tf
from tensorpack import *
import random as rd
import config
import cv2


# compute iou in YXYX format
def compute_iou(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    intersect = max(min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]), 0) \
        * max(min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]), 0)
    return intersect / (area1 + area2 - intersect)


def compute_gtc(gtbbox, bbox):
    area = (gtbbox[2] - gtbbox[0]) * (gtbbox[3] - gtbbox[1])
    intersect = max(min(gtbbox[2], bbox[2]) - max(gtbbox[0], bbox[0]), 0) \
                * max(min(gtbbox[3], bbox[3]) - max(gtbbox[1], bbox[1]), 0)
    return intersect / area


def compute_cd(gtbbox, bbox):
    return np.linalg.norm(np.array([(gtbbox[1] + gtbbox[3]) / 2, (gtbbox[0] + gtbbox[2]) / 2]) -
                          np.array([(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]))


def concat_bbox(bbox, focus):
    w = bbox[3] - bbox[1]
    h = bbox[2] - bbox[0]
    # print(w, h)
    res = [bbox[0] + focus[0] * h, bbox[1] + focus[1] * w,
            bbox[0] + focus[2] * h, bbox[1] + focus[3] * w]
    res = np.clip(res, 0, 1)
    return res


# stage-1 ar
class Env(Callback):
    action_space = [[0, 0, 0.75, 0.75],
                    [0.25, 0, 1, 0.75],
                    [0, 0.25, 0.75, 1],
                    [0.25, 0.25, 1, 1],
                    [0.125, 0.125, 0.875, 0.875],
                    [0.21875, 0, 0.78125, 1],
                    [0, 0.21875, 1, 0.78125],
                    'trigger']

    action_space_refine = [[-0.1, 0, 0.9, 1],
                           [0.1, 0, 1.1, 1],
                           [0, -0.1, 1, 0.9],
                           [0, 0.1, 1, 1.1],
                           'noop']

    def __init__(self, images, bboxes):
        self.images = images
        self.target_bboxes = bboxes
        self.reset()

    def _setup_graph(self):
        pass

    def _before_train(self):
        pass

    def reset(self):
        # focus in normalized coordinate format YXYX
        index = rd.randint(0, len(self.images) - 1)
        self.image = cv2.resize(self.images[index], (224, 224))
        self.target_bbox = self.target_bboxes[index]
        self.focus_image = self.image.copy()
        self.crt_bbox = [0, 0, 1., 1.]
        self.crt_step = 0

        iou = compute_iou(self.target_bbox, self.crt_bbox)
        gtc = compute_gtc(self.target_bbox, self.crt_bbox)
        cd = compute_cd(self.target_bbox, self.crt_bbox)

        self.last_score = iou + gtc + 1 - cd
        self.crt_iou = iou

        self.history = np.zeros([config.HISTORY_LEN, len(self.action_space)], np.float32)

    # focus in YXYX, normalized
    def get_focus(self, focus):
        focus_ymin, focus_ymax = int(focus[0] * self.image.shape[0]), int(focus[2] * self.image.shape[0])
        focus_xmin, focus_xmax = int(focus[1] * self.image.shape[1]), int(focus[3] * self.image.shape[1])
        image = self.focus_image[focus_ymin:focus_ymax, focus_xmin:focus_xmax]
        return cv2.resize(image, (self.image.shape[1], self.image.shape[0]))

    # focus in YXYX, normalized
    def step(self, focus):
        self.crt_step += 1
        if focus == 'trigger':
            iou = compute_iou(self.crt_bbox, self.target_bbox)
            if iou > 0.5:
                return 3, True
            else:
                return -3, True
        else:
            self.focus_image = self.get_focus(focus)
            self.crt_bbox = concat_bbox(self.crt_bbox, focus)
            iou = compute_iou(self.target_bbox, self.crt_bbox)
            gtc = compute_gtc(self.target_bbox, self.crt_bbox)
            cd = compute_cd(self.target_bbox, self.crt_bbox)
            score = iou + gtc + 1 - cd
            ind = 1 if score > self.last_score else -1
            self.crt_iou = iou
            self.last_score = score
            self.history[:-1] = self.history[1:]
            self.history[-1, self.action_space.index(focus)] = 1.

            if self.crt_step >= config.MAX_STEP:
                return ind, True

            return ind, False


from matplotlib import pyplot as plt
if __name__ == '__main__':
    env = Env(None, None)
    env.step([0.33, 0.33, 0.88, 0.88])
    # img = np.array([[[0], [1.]], [[1.], [0.5]]])
    # # img_view = cv2.resize(img, (320, 320), interpolation=cv2.INTER_NEAREST)
    # # cv2.imshow('test', img_view)
    # # cv2.waitKey()
    #
    img = np.arange(200).reshape([10, 10, 2])
    x = tf.placeholder(tf.float32, [None, 10, 10, 2])
    y = tf.image.crop_and_resize(x, [[0.33, 0.33, 0.88, 0.88]], [0], [8, 8])
    with tf.Session() as sess:
        tf_img = sess.run(y, feed_dict={x: img[None, :, :, :]})
        print(tf_img[0, :, :, :])
        # img_view = cv2.resize(tf_img[0], (320, 320), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('test2', img_view)
        # cv2.waitKey()