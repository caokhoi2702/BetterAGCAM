import torch
import torchvision.transforms as transforms
import os
import numpy as np
import gc
import argparse
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.bounding_box import getBoudingBox_multi, box_to_seg
import torch.utils.model_zoo as model_zoo

#datasets
from Datasets.ILSVRC import ImageNetDataset_val

#models
import Methods.AGCAM.ViT_for_AGCAM as ViT_Ours

import timm
from Methods.LRP.ViT_LRP import vit_base_patch16_224 as LRP_vit_base_patch16_224

#methods
from Methods.AGCAM.Better_AGCAM import Better_AGCAM

parser = argparse.ArgumentParser()

MODEL = 'vit_base_patch16_224'
DEVICE = 'cuda'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(777)
if device == DEVICE:
    gc.collect()
    torch.cuda.empty_cache()
print("device: " +  device)
IMG_SIZE=224
THRESHOLD = float(0.5)



# Image transform for ImageNet ILSVRC
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])


validset = ImageNetDataset_val(
    root_dir=args.data_root,
    transforms=transform,
)

# Model Parameter provided by Timm library.
state_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth', progress=True, map_location=device)
class_num=1000

timm_model = timm.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
model = ViT_Ours.create_model(MODEL, pretrained=True, num_classes=class_num).to(device)
model.load_state_dict(state_dict, strict=True)
model.eval()
betterAGC_method = Better_AGCAM(model)


name = "The localization score of " + args.method 



validloader = DataLoader(
    dataset = validset,
    batch_size=1,
    shuffle = False,
)


with torch.enable_grad():        
    num_img = 0
    pixel_acc = 0.0
    dice = 0.0
    precision = 0.0
    recall = 0.0
    iou = 0.0
    for data in tqdm(validloader):
        image = data['image'].to(device)
        label = data['label'].to(device)
        bnd_box = data['bnd_box'].to(device).squeeze(0)

        prediction, heatmaps, output = betterAGC_method.generate(image)
        # If the model produces the wrong predication, the heatmap is unreliable and therefore is excluded from the evaluation.
        if prediction!=label:
            continue
        tensor_heatmaps = heatmaps[0].reshape(144, 1, 14, 14)
        tensor_heatmaps = transforms.Resize((224, 224))(tensor_heatmaps)
        
        min_vals = tensor_heatmaps.amin(dim=(2, 3), keepdim=True)  # Min across width and height
        max_vals = tensor_heatmaps.amax(dim=(2, 3), keepdim=True)  # Max across width and height

        tensor_heatmaps = (tensor_heatmaps - min_vals + 1e-7) / (max_vals - min_vals + 1e-7)

        new_image = torch.mul(tensor_heatmaps, image)
        with torch.no_grad():
            output_mask = timm_model.__call__(new_image)
        
        agc_scores = output_mask[:, prediction.item()] - output[0, prediction.item()]
        agc_scores = torch.sigmoid(agc_scores)
        # agc_scores = torch.softmax(agc_scores, axis = 0)
        agc_scores = agc_scores.reshape(heatmaps[0].shape[0], heatmaps[0].shape[1])


        my_cam = (agc_scores[:, :, None, None, None] * heatmaps[0].to(device)).sum(axis=(0, 1))
        mask = my_cam.unsqueeze(0)
        # my_cam = transforms.Resize((224, 224))(my_cam)
        # my_cam = (my_cam - my_cam.min())/(my_cam.max()-my_cam.min())
        # my_cam = my_cam.detach().cpu().numpy()
        # my_cam = np.transpose(my_cam, (1, 2, 0))
        
        
        mask = mask.reshape(1, 1, 14, 14)
        
    
        # Reshape the mask to have the same size with the original input image (224 x 224)
        upsample = torch.nn.Upsample(224, mode = 'bilinear', align_corners=False)
        mask = upsample(mask)

        # Normalize the heatmap from 0 to 1
        mask = (mask-mask.min())/(mask.max()-mask.min())

        # To avoid the overlapping problem of the bounding box labels, we generate a 0-1 segmentation mask from the bounding box label.
        seg_label = box_to_seg(bnd_box).to(device)

        # From the generated heatmap, we generate a bounding box and then convert it to a segmentation mask to compare with the bounding box label.
        mask_bnd_box = getBoudingBox_multi(mask, threshold=THRESHOLD).to(device)
        seg_mask = box_to_seg(mask_bnd_box).to(device)
        
        output = seg_mask.view(-1, )
        target = seg_label.view(-1, ).float()

        tp = torch.sum(output * target)  # True Positive
        fp = torch.sum(output * (1 - target))  # False Positive
        fn = torch.sum((1 - output) * target)  # False Negative
        tn = torch.sum((1 - output) * (1 - target))  # True Negative
        eps = 1e-5
        pixel_acc_ = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        dice_ = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        precision_ = (tp + eps) / (tp + fp + eps)
        recall_ = (tp + eps) / (tp + fn + eps)
        iou_ = (tp + eps) / (tp + fp + fn + eps)
        
        pixel_acc += pixel_acc_
        dice += dice_
        precision += precision_
        recall += recall_
        iou += iou_
        num_img+=1
             

print(name)
print("result==================================================================")
print("number of images: ", num_img)
print("Threshold: ", THRESHOLD)
print("pixel_acc: {:.4f} ".format((pixel_acc/num_img).item()))
print("iou: {:.4f} ".format((iou/num_img).item()))
print("dice: {:.4f} ".format((dice/num_img).item()))
print("precision: {:.4f} ".format((precision/num_img).item()))
print("recall: {:.4f} ".format((recall/num_img).item()))
