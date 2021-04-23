import cv2
import numpy as np
from itertools import product


def loadImage(img_file):
    img = cv2.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)

    return img

def tlwh_2_maxmin(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        new_bboxes.append([xmin, ymin, xmax, ymax])
    new_bboxes = np.array(new_bboxes)
    return

close_dist = 20
def should_merge(box1, box2, close_dist):
    a = (box1[0], box1[2]), (box1[1], box1[3])
    b = (box2[0], box2[2]), (box2[1], box2[3])

    if any(abs(a_v - b_v) <= close_dist for i in range(2) for a_v, b_v in product(a[i], b[i])):
        return True, [min(*a[0], *b[0]), min(*a[1], *b[1]), max(*a[0], *b[0]), max(*a[1], *b[1])]

    return False, None



def merge_box(boxes, close_dist):
    for i, box1 in enumerate(boxes):
        for j, box2 in enumerate(boxes[i + 1:]):
            is_merge, new_box = should_merge(box1, box2, close_dist)
            if is_merge:
                boxes[i] = None
                boxes[j] = new_box
                break
                
    boxes = [b for b in boxes if b]
    return boxes    

