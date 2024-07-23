import os
import numpy as np
from urllib.request import urlopen
import torchvision.transforms as transforms  
import folder_paths
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps, ImageFilter
import torch.nn as nn
import torch

# comfy_path = os.path.dirname(folder_paths.__file__)
# custom_nodes_path = os.path.join(comfy_path, "custom_nodes")


# 指定本地分割模型文件夹的路径
# model_folder_path = os.path.join(custom_nodes_path,"Comfyui_segformer_b2_clothes","checkpoints","segformer_b3_fashion")
model_folder_path = os.path.join(folder_paths.models_dir,"segformer_b3_fashion")

processor = SegformerImageProcessor.from_pretrained(model_folder_path)
model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 切割服装
def get_segmentation(tensor_image):
    cloth = tensor2pil(tensor_image)
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth


class segformer_b3_fashion:
   
    def __init__(self):
        pass
    
    # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                "image":("IMAGE", {"default": "","multiline": False}), 
                "shirt": ("BOOLEAN", {"default": False, "label_on": "enabled(衬衫、罩衫)", "label_off": "disabled(衬衫、罩衫)"}),
                "top": ("BOOLEAN", {"default": False, "label_on": "enabled(上衣、t恤)", "label_off": "disabled(上衣、t恤)"}),
                "sweater": ("BOOLEAN", {"default": False, "label_on": "enabled(毛衣)", "label_off": "disabled(毛衣)"}),
                "cardigan": ("BOOLEAN", {"default": False, "label_on": "enabled(开襟羊毛衫)", "label_off": "disabled(开襟羊毛衫)"}),
                "jacket": ("BOOLEAN", {"default": False, "label_on": "enabled(夹克)", "label_off": "disabled(夹克)"}),
                "vest": ("BOOLEAN", {"default": False, "label_on": "enabled(背心)", "label_off": "disabled(背心)"}),
                "pants": ("BOOLEAN", {"default": False, "label_on": "enabled(裤子)", "label_off": "disabled(裤子)"}),
                "shorts": ("BOOLEAN", {"default": False, "label_on": "enabled(短裤)", "label_off": "disabled(短裤)"}),
                "skirt": ("BOOLEAN", {"default": False, "label_on": "enabled(裙子)", "label_off": "disabled(裙子)"}),
                "coat": ("BOOLEAN", {"default": False, "label_on": "enabled(外套)", "label_off": "disabled(外套)"}),
                "dress": ("BOOLEAN", {"default": False, "label_on": "enabled(连衣裙)", "label_off": "disabled(连衣裙)"}),
                "jumpsuit": ("BOOLEAN", {"default": False, "label_on": "enabled(连身裤)", "label_off": "disabled(连身裤)"}),
                "cape": ("BOOLEAN", {"default": False, "label_on": "enabled(斗篷)", "label_off": "disabled(斗篷)"}),
                "glasses": ("BOOLEAN", {"default": False, "label_on": "enabled(眼镜)", "label_off": "disabled(眼镜)"}),
                "hat": ("BOOLEAN", {"default": False, "label_on": "enabled(帽子)", "label_off": "disabled(帽子)"}),
                "hairaccessory": ("BOOLEAN", {"default": False, "label_on": "enabled(头带)", "label_off": "disabled(头带)"}),
                "tie": ("BOOLEAN", {"default": False, "label_on": "enabled(领带)", "label_off": "disabled(领带)"}),
                "glove": ("BOOLEAN", {"default": False, "label_on": "enabled(手套)", "label_off": "disabled(手套)"}),
                "watch": ("BOOLEAN", {"default": False, "label_on": "enabled(手表)", "label_off": "disabled(手表)"}),
                "belt": ("BOOLEAN", {"default": False, "label_on": "enabled(皮带)", "label_off": "disabled(皮带)"}),
                "legwarmer": ("BOOLEAN", {"default": False, "label_on": "enabled(暖腿器)", "label_off": "disabled(暖腿器)"}),
                "tights": ("BOOLEAN", {"default": False, "label_on": "enabled(紧身衣、长筒袜)", "label_off": "disabled(紧身衣、长筒袜)"}),
                "sock": ("BOOLEAN", {"default": False, "label_on": "enabled(袜子)", "label_off": "disabled(袜子)"}),
                "shoe": ("BOOLEAN", {"default": False, "label_on": "enabled(鞋子)", "label_off": "disabled(鞋子)"}),
                "bagwallet": ("BOOLEAN", {"default": False, "label_on": "enabled(包、钱包)", "label_off": "disabled(包、钱包)"}),
                "scarf": ("BOOLEAN", {"default": False, "label_on": "enabled(围巾)", "label_off": "disabled(围巾)"}),
                "umbrella": ("BOOLEAN", {"default": False, "label_on": "enabled(雨伞)", "label_off": "disabled(雨伞)"}),
                "hood": ("BOOLEAN", {"default": False, "label_on": "enabled(兜帽)", "label_off": "disabled(兜帽)"}),
                "collar": ("BOOLEAN", {"default": False, "label_on": "enabled(衣领)", "label_off": "disabled(衣领)"}),
                "lapel": ("BOOLEAN", {"default": False, "label_on": "enabled(翻领)", "label_off": "disabled(翻领)"}),
                "epaulette": ("BOOLEAN", {"default": False, "label_on": "enabled(肩章)", "label_off": "disabled(肩章)"}),
                "sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled(袖子)", "label_off": "disabled(袖子)"}),
                "pocket": ("BOOLEAN", {"default": False, "label_on": "enabled(口袋)", "label_off": "disabled(口袋)"}),
                "neckline": ("BOOLEAN", {"default": False, "label_on": "enabled(领口)", "label_off": "disabled(领口)"}),
                "buckle": ("BOOLEAN", {"default": False, "label_on": "enabled(带扣)", "label_off": "disabled(带扣)"}),
                "zipper": ("BOOLEAN", {"default": False, "label_on": "enabled(拉链)", "label_off": "disabled(拉链)"}),
                "applique": ("BOOLEAN", {"default": False, "label_on": "enabled(贴花)", "label_off": "disabled(贴花)"}),
                "bead": ("BOOLEAN", {"default": False, "label_on": "enabled(珠子)", "label_off": "disabled(珠子)"}),
                "bow": ("BOOLEAN", {"default": False, "label_on": "enabled(蝴蝶结)", "label_off": "disabled(蝴蝶结)"}),
                "flower": ("BOOLEAN", {"default": False, "label_on": "enabled(花)", "label_off": "disabled(花)"}),
                "fringe": ("BOOLEAN", {"default": False, "label_on": "enabled(额前短垂发)", "label_off": "disabled(额前短垂发)"}),
                "ribbon": ("BOOLEAN", {"default": False, "label_on": "enabled(丝带)", "label_off": "disabled(丝带)"}),
                "rivet": ("BOOLEAN", {"default": False, "label_on": "enabled(铆钉)", "label_off": "disabled(铆钉)"}),
                "ruffle": ("BOOLEAN", {"default": False, "label_on": "enabled(褶饰)", "label_off": "disabled(褶饰)"}),
                "sequin": ("BOOLEAN", {"default": False, "label_on": "enabled(亮片)", "label_off": "disabled(亮片)"}),
                "tassel": ("BOOLEAN", {"default": False, "label_on": "enabled(流苏)", "label_off": "disabled(流苏)"}),
                }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_image",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,image,
shirt,
top,
sweater,
cardigan,
jacket,
vest,
pants,
shorts,
skirt,
coat,
dress,
jumpsuit,
cape,
glasses,
hat,
hairaccessory,
tie,
glove,
watch,
belt,
legwarmer,
tights,
sock,
shoe,
bagwallet,
scarf,
umbrella,
hood,
collar,
lapel,
epaulette,
sleeve,
pocket,
neckline,
buckle,
zipper,
applique,
bead,
bow,
flower,
fringe,
ribbon,
rivet,
ruffle,
sequin,
tassel):
        
        results = []
        for item in image:
        
            # seg切割结果，衣服pil
            pred_seg,cloth = get_segmentation(item)
            labels_to_keep = [0]
            # if background :
            #     labels_to_keep.append(0)
            if not shirt:
                labels_to_keep.append(1)
            if not top:
                labels_to_keep.append(2)
            if not sweater:
                labels_to_keep.append(3)
            if not cardigan:
                labels_to_keep.append(4)
            if not jacket:
                labels_to_keep.append(5)
            if not vest:
                labels_to_keep.append(6)
            if not pants:
                labels_to_keep.append(7)
            if not shorts:
                labels_to_keep.append(8)
            if not skirt:
                labels_to_keep.append(9)
            if not coat:
                labels_to_keep.append(10)
            if not dress:
                labels_to_keep.append(11)
            if not jumpsuit:
                labels_to_keep.append(12)
            if not cape:
                labels_to_keep.append(13)
            if not glasses:
                labels_to_keep.append(14)
            if not hat:
                labels_to_keep.append(15)
            if not hairaccessory:
                labels_to_keep.append(16)
            if not tie:
                labels_to_keep.append(17)
            if not glove:
                labels_to_keep.append(18)
            if not watch:
                labels_to_keep.append(19)
            if not belt:
                labels_to_keep.append(20)
            if not legwarmer:
                labels_to_keep.append(21)
            if not tights:
                labels_to_keep.append(22)
            if not sock:
                labels_to_keep.append(23)
            if not shoe:
                labels_to_keep.append(24)
            if not bagwallet:
                labels_to_keep.append(25)
            if not scarf:
                labels_to_keep.append(26)
            if not umbrella:
                labels_to_keep.append(27)
            if not hood:
                labels_to_keep.append(28)
            if not collar:
                labels_to_keep.append(29)
            if not lapel:
                labels_to_keep.append(30)
            if not epaulette:
                labels_to_keep.append(31)
            if not sleeve:
                labels_to_keep.append(32)
            if not pocket:
                labels_to_keep.append(33)
            if not neckline:
                labels_to_keep.append(34)
            if not buckle:
                labels_to_keep.append(35)
            if not zipper:
                labels_to_keep.append(36)
            if not applique:
                labels_to_keep.append(37)
            if not bead:
                labels_to_keep.append(38)
            if not bow:
                labels_to_keep.append(39)
            if not flower:
                labels_to_keep.append(40)
            if not fringe:
                labels_to_keep.append(41)
            if not ribbon:
                labels_to_keep.append(42)
            if not rivet:
                labels_to_keep.append(43)
            if not ruffle:
                labels_to_keep.append(44)
            if not sequin:
                labels_to_keep.append(45)
            if not tassel:
                labels_to_keep.append(46)
                
            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            
            # 创建agnostic-mask图像
            mask_image = Image.fromarray(mask * 255)
            mask_image = mask_image.convert("RGB")
            mask_image = pil2tensor(mask_image)
            results.append(mask_image)

        return (torch.cat(results, dim=0),)