import cv2
import os 
import time

import numpy as np

from .utils import get_config, loadImage, sorting_bounding_box, visual, align_item, tlwh_2_maxmin, merge_box

from .libs.CRAFT.craft import CRAFT
from .libs.MORAN.MORAN_pred import MORAN_predict
from .libs.MORAN.models.moran import MORAN
from .libs.DeepText.Deeptext_pred import Deeptext_predict, load_model_Deeptext
# from .libs.detectron2.predict_img import predict_img_detectron2
# from .libs.detectron2.predict_img import visualize
from .libs.super_resolution.improve_resolution import improve_resolution
from .libs.CRAFT.craft_predict import craft_text_detect, load_model_Craft

from .src import yolo_detect

# setup config
cfg = get_config()
cfg.merge_from_file('./app/LP_reg_src/configs/pipeline.yaml')
cfg.merge_from_file('./app/LP_reg_src/configs/craft.yaml')
cfg.merge_from_file('./app/LP_reg_src/configs/faster.yaml')
cfg.merge_from_file('./app/LP_reg_src/configs/yolo.yaml')


CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()
PIPELINE_CFG = cfg.PIPELINE

# load all model
# model yolo
print ('[LOADING] Detect model')
YOLO_NET = cv2.dnn.readNet(cfg.YOLOV4.YOLO_MODEL_PATH, cfg.YOLOV4.YOLO_CFG_PATH)
print ('[LOADING SUCESS] Detect model')
# model text detct
print ('[LOADING] Text detecttion model')
CRAFT_MODEL = load_model_Craft(CRAFT_CONFIG, NET_CRAFT)
print ('[LOADING SUCESS] Text detection model')
# model regconition
print ('[LOADING] Text regconition model')
DEEPTEXT_MODEL, DEEPTEXT_PREDICTION, DEEPTEXT_CONVERTER = load_model_Deeptext(cfg.PIPELINE.DEEPTEXT_MODEL_PATH)
print ('[LOADING SUCESS] Text regconition model')

def text_recog(cfg, image_path, model, Prediction, converter):
    text = 'None'
    if cfg.PIPELINE.DEEPTEXT:
        list_image_path = [image_path]
        for img in list_image_path:
            text = Deeptext_predict(img, model, Prediction, converter)
    elif cfg.PIPELINE.MORAN:
        text = MORAN_predict(cfg.PIPELINE.MORAN_MODEL_PATH, image_path, MORAN)
    return text

def text_detect_CRAFT(img, craft_config, CRAFT_MODEL, sortbb=True, visual_img=False):
    # img = loadImage(image_path)
    bboxes, polys, score_text = craft_text_detect(img, craft_config, CRAFT_MODEL)

    if sortbb:
        polys = sorting_bounding_box(polys)
    if visual_img:
        img = visual(img, polys)

    return bboxes, polys, score_text

def LP_detect_faster(img, cfg):
    classes = ['LP']
    outputs = predict_img_detectron2(cfg.FASTER_RCNN.MODEL, cfg.FASTER_RCNN.CONFIG, cfg.FASTER_RCNN.CONFIDENCE_THRESHOLD, cfg.FASTER_RCNN.NUM_OF_CLASS, img)
    frame = visualize (outputs, img, classes)
    cv2.imwrite('frame.jpg', frame)
    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes
    return boxes


def LP_detect_yolo(img, cfg, YOLO_NET):
    img, class_ids, boxes = yolo_detect(img, YOLO_NET, cfg)
    return boxes


def LP_regconition(cfg, img, YOLO_NET):
    
    # detect License plates in image    
    detected_LP = LP_detect_yolo(img, cfg, YOLO_NET)
    for i in detected_LP:
        # store the license plate in image to new_img variable
        print ("detected license plates: ", i)
        new_img = img[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
        cv2.imwrite('./result/LP.jpg', new_img)

        # predict region of text bounding box
        bboxes, polys, score_text = text_detect_CRAFT(new_img, CRAFT_CONFIG, CRAFT_MODEL)
        LP_reg = []

        for index, bbox in enumerate(bboxes):
            # merge bbox on a line
            img_reg = new_img[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]
            img_reg = improve_resolution(img_reg)
            cv2.imwrite('./LP_reg_src/reg/img_reg.jpg', img_reg)
            text = text_recog (cfg, './LP_reg_src/reg/img_reg.jpg', DEEPTEXT_MODEL, DEEPTEXT_PREDICTION, DEEPTEXT_CONVERTER)
            LP_reg.append(text)
            cv2.rectangle(new_img, (bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]), (0,255,0), 1)
            # cv2.putText(new_img, str(count), (bbox[0][0], bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1)
        LP_reg_text = ''.join(LP_reg)
        LP_reg_text = LP_reg_text.upper()
        cv2.putText(img, str(LP_reg_text), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255,255,0), thickness=3)
    return img

if __name__ == '__main__':
    # start = time.time()
    # path = './data/reg_data'
    # save = './result_text_detect/'
    # detect_on_image(cfg, path)
    # for i in os.listdir(path):
    #     path_save = os.path.join(save, i)
    #     img_path = os.path.join(path, i)
    #     print (path_save)
    #     img = cv2.imread(img_path)
    #     bboxes, polys, score_text = text_detect_CRAFT(img, CRAFT_CONFIG, NET_CRAFT, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
    #     for i in bboxes:
    #         cv2.rectangle(img, (int (i[0][0]), int(i[0][1])), (int (i[2][0]), int(i[2][1])), (0,255,255), 1)
    #     cv2.imwrite(path_save, img)

    img = cv2.imread('./data/LP_8539.jpg')
    start = time.time()
    img = LP_regconition(cfg, img, YOLO_NET)
    end = time.time()
    print (np.abs (start - end))
    cv2.imwrite('./result/8539.jpg', img)

    # img = cv2.imread('data/check.png')
    # bboxes, polys, score_text = text_detect_CRAFT(img, CRAFT_CONFIG, NET_CRAFT, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
    # for i in bboxes:
    #     cv2.rectangle(img, (int (i[0][0]), int(i[0][1])), (int (i[2][0]), int(i[2][1])), (255,0,0), 2)
    # cv2.imwrite('Khang.jpg', img)
