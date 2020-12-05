import json
from detectron2.engine import DefaultPredictor
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
import json
import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from utils import binary_mask_to_rle


def test(cfg):
    predictor = DefaultPredictor(cfg)
    ret = []

    df = json.load(open('dataset/test.json'))

    # create categories dictionary
    cats = df['categories']
    cat_dict = {}
    for cat in cats:
        cat_dict[cat['name']] = cat['id']

    for image in df['images']:
        file_name = image['file_name']
        height = image['height']
        width = image['width']
        print("testing ", file_name)
        file_name = os.path.join("dataset/test_images", file_name)
        image_id = image['id']
        img = cv2.imread(file_name)
        output = predictor(img)
        info = output['instances'].get_fields()
        num_inst = len(info['scores'].to("cpu"))
        for i in range(num_inst):
            instance = {}
            instance['image_id'] = image_id
            instance['score'] = float(info['scores'][i].to("cpu").numpy())
            instance['category_id'] = int(
                info['pred_classes'][i].to("cpu").numpy()+1)

            bit_mask = info['pred_masks'][i].to("cpu").numpy()
            rle = binary_mask_to_rle(bit_mask)
            rle['size'] = [height, width]
            instance['segmentation'] = rle

            ret.append(instance)

            # print(type(instance['image_id']))
            # print(type(instance['score']))
            # print(type(instance['category_id']))
            # print(type(instance['segmentation']))
            # print(type(instance['segmentation']['size']))
            # print(type(instance['segmentation']['counts']))

    return ret


def test_one(cfg):
    file_name = os.path.join("dataset/test_images", '2011_002641.jpg')
    img = cv2.imread(file_name)
    output = predictor(img)

    bit_mask = output['instances'].get_fields()['pred_masks'].to("cpu").numpy()
    rle = binary_mask_to_rle(bit_mask)
    print(rle)

    print(output['instances'].get_fields()['scores'][0].to("cpu").numpy())

    v = Visualizer(img[:, :, ::-1],
                   # metadata=balloon_metadata,
                   scale=1.0,
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   instance_mode=ColorMode.IMAGE_BW
                   )
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv2.imwrite("test.jpg", out.get_image()[:, :, ::-1])
