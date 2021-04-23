import cv2
import os 
import time

import numpy as np

# from utils import get_config, loadImage, sorting_bounding_box, visual, align_item, tlwh_2_maxmin, merge_box

# from libs.CRAFT.craft import CRAFT
# from libs.MORAN.MORAN_pred import MORAN_predict
# from libs.MORAN.models.moran import MORAN
from libs.DeepText.Deeptext_pred import Deeptext_predict, load_model_Deeptext
from libs.super_resolution.improve_resolution import improve_resolution

from src import craft_text_detect, load_model_Craft
from src import yolo_detect

img = cv2.imread('./result.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = cv2.equalizeHist(img)
ret, thresh1 = cv2.threshold(img,175,255,cv2.THRESH_BINARY)


cv2.imwrite('./result_thresh.jpg', thresh1)
# cv2.waitKey(0)