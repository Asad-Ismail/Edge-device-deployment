from ast import arg
from d2go.model_zoo import model_zoo
import cv2
from matplotlib import pyplot as plt
from d2go.utils.demo_predictor import DemoPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import json
import numpy as np
from detectron2.structures import BoxMode
from d2go.runner import GeneralizedRCNNRunner
import torch
import copy
from detectron2.data import build_detection_test_loader
from d2go.export.api import convert_and_export_predictor
from d2go.utils.testing.data_loader_helper import create_fake_detection_data_loader
from d2go.export.d2_meta_arch import patch_d2_meta_arch
from detectron2.utils.logger import setup_logger
from mobile_cv.predictor.api import create_predictor
import logging
from typing import List, Dict
import detectron2.data.transforms as T
import math
import argparse
setup_logger()

# Valid image extensions
extensions = [".jpg", ".JPEG", ".jpg", ".jpeg", ".png"]


def preprocess_input(original_image):
    """Preporcess Input Inmages for model inferece

    Args:
        original_image ([np.array]): Original Image

    Returns:
        [List[Tensor]]: Transformed float tensor
    """
    min_size_test=224
    max_size_test=320
    aug = T.ResizeShortestEdge([min_size_test, min_size_test], max_size_test)
    image = aug.get_transform(original_image).apply_image(original_image)
    #image=original_image
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image/=255
    return [image]
            
class Wrapper(torch.nn.Module):
    """Wrapper on original model to do preprocessing and return results for torch scripted model

    """
    def __init__(self, model):
        """
        Args:
            model ([nn.Module]): Original Model to be wrapped
        """
        super().__init__()
        self.model = model
        coco_idx_list = [1]
        self.coco_idx = torch.tensor(coco_idx_list)

    def forward(self, inputs: List[torch.Tensor]):
        x = inputs[0].unsqueeze(0) * 255
        scale = 320.0 / min(x.shape[-2], x.shape[-1])
        x = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True, recompute_scale_factor=True)
        out = self.model(x[0])
        res : Dict[str, torch.Tensor] = {}
        #res_mask={}
        res["boxes"] = out[0] / scale
        res["labels"] = torch.index_select(self.coco_idx, 0, out[1])
        res["scores"] = out[3]
        #res["masks"]=out[2]
        boxes_dims=res["boxes"][:,2:]-res["boxes"][:,0:2]
        for i in range(boxes_dims.shape[0]):
            mask_i=out[2][i].unsqueeze(0)
            resize_i=tuple((int(boxes_dims[i][1]),int(boxes_dims[i][0])))
            inter_m = torch.nn.functional.interpolate(mask_i, size=resize_i, mode="bilinear",align_corners=True)
            res[str(i)]=inter_m
            #res[str(i)]=mask_i.squeeze()
        #res_mask["masks"]=inter_masks
        return inputs, [res]


def prepare_for_launch(veg,model_type):
    """Get Config and trainer 

    Args:
        veg ([str]): Name of dataset
        model_type ([str]): Type of model FasterRCNN/MaskRCNN e.tc

    Returns:
        [Tuple]: config and Trainer
    """
    runner = GeneralizedRCNNRunner()
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_type))
    cfg.MODEL_EMA.ENABLED = False
    cfg.DATASETS.TRAIN = (veg+"_train",)
    cfg.DATASETS.TEST = (veg+"_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_type)  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS="/home/asad/projs/d2go/output/model_final.pth"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 20000    # 600 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.QUANTIZATION.BACKEND='qnnpack'
    #cfg.QUANTIZATION.EAGER_MODE=False
    return cfg, runner


def get_veg_dicts(img_dir):
    """Get Vegetable dataset

    Args:
        img_dir ([str]): Directory with images and annotation in json format

    Returns:
        [Dict]: Dataset Dictonary
    """
    json_files = [
        json_file
        for json_file in os.listdir(img_dir)
        if json_file.endswith(".json")
    ]
    dataset_dicts = []
    for idx, json_file in enumerate(json_files):
        for ext in extensions:
            filename = json_file.split(".")[0] + ext
            c_fname = os.path.join(img_dir, filename)
            img = cv2.imread(c_fname)
            if img is not None:
                break
        if img is None:
            print(f"Image Not Found for {json_file}")
            raise (f"Image Not Found for {json_file}")
        with open(os.path.join(img_dir, json_file)) as f:
            imgs_anns = json.load(f)
        record = {}
        height, width = img.shape[:2]
        record["file_name"] = c_fname
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = imgs_anns["shapes"]
        objs = []
        for anno in annos:
            # assert not anno["region_attributes"]
            px = [x for x, y in anno]
            py = [y for x, y in anno]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        print(f"Processed images {idx}")
        dataset_dicts.append(record)
    return dataset_dicts


def resize_masks(pixels,w1, h1, w2, h2):
    """Resize Image using nearest Inprepolation. Used in Android Version of D2Go we can also use opencv version here

    Args:
        pixels ([np.array]): h1*w1 image array
        w1 ([int]): Current Width
        h1 ([int]): Current Height
        w2 ([int]): Output Width
        h2 ([int]): Output Height

    Returns:
        [np.array]: No array of size h2xw2
    """
    pixels=pixels.flatten()
    temp= np.zeros(w2*h2)
    x_ratio = w1/float(w2) 
    y_ratio = h1/float(h2) 
    for i in range(h2):
        for j in range(w2):
            px = math.floor(j*x_ratio)
            py = math.floor(i*y_ratio) 
            temp[(i*w2)+j] = pixels[(int)((py*w1)+px)] 
    temp=temp.reshape((h2,w2))
    return temp


def vis_mask_rect(img,rect,mask):
    """Visualize MaskRCNN Mask Without resizing them to original reolution

    Args:
        img ([np.array]): Input Image flatten array
        rect ([Tuple]): Tuple of rectangle
        mask ([np.array]): Mask to be drawn on image
    """
    x1,y1,x2,y2=rect
    for j in range(y1,y2-1):
        for i in range(x1,x2-1):
            if mask[j-y1,i-x1]>0.5:
                cv2.circle(img,(i,j),1,(0,0,255))
        
        
def vis_results(image,results):
    """Viusalize MasKRCNN results

    Args:
        image ([np.array]): HxWx3 Input image
        results ([dict]): Result dict
    """
    x = image[0] * 255
    x=x.permute(1,2,0).to("cpu").numpy().astype(np.uint8)
    #vis_copy=x.copy()
    for index in range(len(results[0]["boxes"])):
        box=results[0]["boxes"][index].to("cpu").numpy()
        mask=results[0][str(index)].to("cpu").squeeze().numpy().squeeze()
        boxW=int(box[2]-box[0])
        boxH=int(box[3]-box[1])
        mask=resize_masks(mask,mask.shape[1],mask.shape[0],boxW,boxH)
        rect=int(box[0]),int(box[1]),int(box[2]),int(box[3])
        cv2.rectangle(x,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
        vis_mask_rect(x,rect,mask)
    plt.imshow(x[:, :, ::-1])
    plt.title("Interpolation visualization")
    plt.show()
    
    
if __name__ =="__main__":
    
    argparser=argparse.ArgumentParser()
    argparser.add_argument("--train",default=False,help="To Train the network")
    argparser.add_argument("--network",default="qat_mask_rcnn_fbnetv3a_C4.yaml",help="Dataset directory for Training")
    argparser.add_argument("--val",default=False,help="To Validate the results and visualize")
    argparser.add_argument("--m_conversion",default=True,help="Convert Network to Mobile format")
    argparser.add_argument("--data_dir",default="/media/asad/ADAS_CV/datasets_Vegs/pepper/annotated/scalecam1_2",help="Dataset directory for Training")
    argparser.add_argument("--img",default="images/test.jpg",help="Image for validaiton")
    
    args=argparser.parse_args()
    # Patch for QAT to original Detectron2 Models
    patch_d2_meta_arch()
    # For arm processors
    torch.backends.quantized.engine = 'qnnpack'
    # Dataset name
    veg="pepp"
    # Set Up Data and evaluator
    for d in ["train", "valid"]:
        DatasetCatalog.register(veg + "_" + d, lambda d=d: get_veg_dicts(os.path.join(args.data_dir, d)))
        MetadataCatalog.get(veg + "_" + d).set(thing_classes=[veg], evaluator_type="coco")
    veg_metadata = MetadataCatalog.get(veg+"_train")
    # Set up configs and model
    cfg, runner = prepare_for_launch(veg,args.network)
    model = runner.build_model(cfg)
    if args.train:
        runner.do_train(cfg, model, resume=False)
    if args.val:
        img=args.img
        im=cv2.imread(img)
        state_dict=torch.load("output/model_final.pth")["model"]
        model.load_state_dict(state_dict=state_dict,strict=False)
        metrics = runner.do_test(cfg, model)
        print(metrics)
        predictor = DemoPredictor(model)
        outputs = predictor(im)
        v = Visualizer(im, veg_metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.title("Original Model Validation")
        plt.show()
    if args.m_conversion:
        previous_level = logging.root.manager.disable
        logging.disable(logging.INFO)
        img=args.img
        im=cv2.imread(img)
        state_dict=torch.load("output/model_final.pth")["model"]
        model.load_state_dict(state_dict=state_dict,strict=False)
        pytorch_model=model
        pytorch_model.to("cpu").eval()
        with create_fake_detection_data_loader(224, 320, is_train=False) as data_loader:
            predictor_path = convert_and_export_predictor(cfg,copy.deepcopy(pytorch_model),"torchscript_int8",'./',data_loader,)
        orig_model = torch.jit.load(os.path.join(predictor_path, "model.jit"))
        wrapped_model = Wrapper(orig_model)
        # optionally do a forward
        inp=preprocess_input(im)
        inp,test_results=wrapped_model(inp)
        vis_results(inp,test_results)
        scripted_model = torch.jit.script(wrapped_model)
        scripted_model.save("d2go.pt") 
        logging.disable(previous_level)
        # Visualize Converted Data
        model = create_predictor(predictor_path)
        predictor = DemoPredictor(model)
        outputs = predictor(im)
        v = Visualizer(im, veg_metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.title("Quantized Model")
        plt.show()
        
    
        
