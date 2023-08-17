import torch
from pytorch_msssim import ssim,ms_ssim, SSIM, MS_SSIM
from torchmetrics import MeanSquaredError
from PIL import Image
import numpy as np
import os


def eval_loss(model, loader, segment_criterion, test_or_val = True):
    renormalized_list=[]
    model.eval()
    model=model.cuda()
    seg_loss=0
    counter = 0
    ssim_total = []

    for i, (images,heatmaps,file_name) in enumerate(loader):
        images = images.float().cuda()
        heatmaps = heatmaps.cuda()
        
        l_segment = 0
        with torch.no_grad():
            masks_pred = model(images)
            l_segment = segment_criterion(masks_pred,heatmaps.float())
            seg_loss += l_segment
            counter += 1
            my_ssim=ssim(heatmaps[:,:,19:30,19:30].float(),masks_pred[:, :,19:30,19:30])
            mean_squared_error = MeanSquaredError()
            target_cropped=heatmaps[:,:,19:30,19:30].detach().cpu()
            predicted_cropped=masks_pred[:, :,19:30,19:30].detach().cpu()
            mse=mean_squared_error(predicted_cropped, target_cropped)
            my_max_diff=abs(torch.max(masks_pred[:, :,19:30,19:30])-torch.max( heatmaps[:,:,19:30,19:30]))
        

        #We need to change the model name here as well as changing output model path in train
        if test_or_val == True:
                pil_image = Image.fromarray(masks_pred.detach().cpu().numpy().astype(np.uint8)[0, 0])                
                # path = f"E:\\Documents\\MRgLITT\\data\\500J\\fixed_outputs\\heatmap_predicted\\{i}.png"
                path = f"E:\Documents\MRgLITT\data\\500J\original\preprocessed\\both_pre\\fixed_outputs\heatmap_predicted\\{file_name[0][:-4]}.png"
                # path = f"E:\\MRgLITT Data\\new data\\2021-12-21\\2_more_data_sub-folders_MTLE\\PNG\\500J\\S2\\fixed_outputs_augmented\\heatmap_predicted\\{file_name[0][:-4]}.png"
                pil_image.save(path)

                pil_image = Image.fromarray(images.detach().cpu().numpy().astype(np.uint8)[0, 0])                
                # path = f"E:\\Documents\\MRgLITT\\data\\500J\\fixed_outputs_augmented\\MRI\\{i}.png"
                path = f"E:\Documents\MRgLITT\data\\500J\original\preprocessed\\both_pre\\fixed_outputs\MRI\\{file_name[0][:-4]}.png"
                # path = f"E:\\MRgLITT Data\\new data\\2021-12-21\\2_more_data_sub-folders_MTLE\\PNG\\500J\\S2\\fixed_outputs_augmented\\MRI\\{file_name[0][:-4]}.png"
                pil_image.save(path)

                pil_image = Image.fromarray(heatmaps.detach().cpu().numpy().astype(np.uint8)[0, 0])                
                # path = f"E:\\Documents\\MRgLITT\\data\\500J\\fixed_outputs_augmented\\heatmap_ground_truth\\{i}.png"
                path = f"E:\Documents\MRgLITT\data\\500J\original\preprocessed\\both_pre\\fixed_outputs\heatmap_ground_truth\\{file_name[0][:-4]}.png"
                # path = f"E:\\MRgLITT Data\\new data\\2021-12-21\\2_more_data_sub-folders_MTLE\\PNG\\500J\\S2\\fixed_outputs\\heatmap_ground_truth\\{file_name[0][:-4]}.png"
                pil_image.save(path)
    return l_segment,my_ssim,my_max_diff,mse