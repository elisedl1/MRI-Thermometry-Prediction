import Project_Dataloader_cv
from Project_Dataloader_cv import  MRIDataset

from model import UNet, regression_UNet, AttU_Net
from  evaluation import eval_loss
from torch import optim
from torch import nn
import argparse
from torch.utils.data import DataLoader,SubsetRandomSampler
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import KFold
import random
from copy import deepcopy

from PIL import Image
from pytorch_msssim import ssim
from torchmetrics import MeanSquaredError

import datetime
import csv

import logging


def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Eye Gaze UNet')

    # Data
    parser.add_argument('--image_path', type=str, default="", help='image_path')
    parser.add_argument('--heatmaps_path', type=str, help='Heatmaps directory', default='')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    # parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0022, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')


    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser

def train_network(args,model,data_ds,image_path,thermometry_path): 

    # Collect Logger Info for model parameters
    logging.info('Starting train_network()...')
    logging.info('Energy Level: 500J - Epochs: {} - Learning Rate: {} - Batch Size: {} - Drop Out: {}'.format(args.epochs, args.lr, args.batch_size, args.dropout))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    segmentation_loss = nn.functional.l1_loss#nn.BCEWithLogitsLoss()# nn.CrossEntropyLoss()#GeneralizedDiceLoss()
    epoch_loss_list = []
    fold_train_loss=[]
    fold_valid_loss=[]
    val_ssim_list=[]
    n_parameters=[]
    max_list=[]
    mse_list=[]
    max_ssim=0    
    #For nested cross validation to seperate test ids from development (train and validatrion)
    for test_fold, (tr_idx,test_idx) in enumerate(splits.split(np.arange(len(data_ds)))): #we can decrease it if it takes a lot
        tr_sampler = SubsetRandomSampler(tr_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        test_list=[data_ds[j] for j in test_idx]    
        # test_ds=MRIDataset_im(test_list,image_path,thermometry_path)      
        test_ds=MRIDataset(test_list,image_path,thermometry_path,"image_heatmap")
        print("testtttttttttttt dsssssssss", type(test_ds))
        # print("test_ds",len(test_ds))
        test_dl = DataLoader(test_ds)#, sampler=test_sampler) #problem caused by test_sampler

        development_ds=[data_ds[j] for j in tr_idx] #combination od train and validation
        print('Test Fold {}'.format(test_fold + 1))

        # if test_fold ==5:
        #     break

        #Inside cross validation for train and validation
        for valid_fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(development_ds)))):
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(val_idx)
            train_list=[development_ds[j] for j in train_idx]
            valid_list=[development_ds[j] for j in val_idx]
            train_ds=MRIDataset(train_list,image_path,thermometry_path,"image_heatmap")
            print("train_ds",type(train_ds))
            valid_ds=MRIDataset(valid_list,image_path,thermometry_path,"image_heatmap")
            train_dl =DataLoader(train_ds, batch_size=args.batch_size)  #in train_dl we have batch sized data  
            valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)
    
            fold_train_loss.append([])
            fold_valid_loss.append([])
            print('Valid Fold {}'.format(valid_fold + 1))

            model=UNet()#regression_UNet()#UNet()#AttU_Net()#UNet()#AttU_Net()#UNet()
            print("n_parameters",sum(p.numel() for p in model.parameters() if p.requires_grad))
            model=model.cuda()


            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            optimizer.zero_grad()
            for epoch in range(args.epochs):
                model.train()
                model=model.float().cuda()
                counter = 0
                fold_ssim_list = [] # ssim list for particular fold
                # print("train dataloader",train_ds.__len__)
                for images, heatmaps,file_name in (train_dl):
                        
                        images=images.float().cuda()
                        heatmaps = heatmaps.cuda()
                        masks_pred= model(images)
                    
                        loss_segment = segmentation_loss(masks_pred, heatmaps.float()) #to read what exactly it is
                        #epoch_loss += loss_segment.item()

                        optimizer.zero_grad()
                        loss_segment.backward()
                        # -- clip the gradient at a specified value in the range of [-clip, clip].
                        # -- This is mainly used to prevent exploding or vanishing gradients in the network training
                        nn.utils.clip_grad_value_(model.parameters(), 0.1)
                        optimizer.step()
                #     global_step += 1
                        counter += 1

                with torch.no_grad():
                            
                    val_segment,ss,val_max,val_mse = eval_loss(model, valid_dl, segmentation_loss, False)    

                                            
                    #print("validation_loss",val_segment.item())
                    #print("validation_ssim",ss)
                    print("epoch", epoch, "| validation SSIM:", ss.item())
                    val_ssim_list.append(ss)
                    mse_list.append(val_mse)
                    max_list.append(val_max)

                    #add fold ss to fold ssim list
                    fold_ssim_list.append(ss)

                    if ss>=max_ssim:
                        max_ssim=ss

                        # deep copy save of current best model
                        best_weights = deepcopy(model.state_dict())
                      
                #epoch_loss_list.append(epoch_loss)
                fold_train_loss[valid_fold].append(loss_segment.item())
                fold_valid_loss[valid_fold].append(val_segment.item())               

            if epoch==499:
                torch.save(best_weights, output_model_path + f"/test_fold{test_fold}_valid_fold{valid_fold}_Epoch_{epoch+1}.pth")

                mean_fold_ssim = torch.mean(torch.stack(fold_ssim_list))

                #torch.save(model.state_dict(), output_model_path + f"/test_fold{test_fold}_valid_fold{valid_fold}_Epoch_{epoch+1}.pth")
                print("train_loss_list:",fold_train_loss)
                print("validation_loss_list",fold_valid_loss)

                logging.info(" ")
                logging.info('Add Evaluation for Valid Fold {}...'.format(valid_fold + 1))                           
                #mean_train_loss = torch.mean(torch.tensor(fold_train_loss))
                #mean_valid_loss = torch.mean(torch.tensor(fold_valid_loss))
                logging.info("mean_val_ssim: {}".format(mean_fold_ssim))
                    
        print("mean_val_ssim",torch.mean(torch.stack(val_ssim_list)))
        print("mean_val_mse",torch.mean(torch.stack(mse_list)))
        print("mean_val_max",torch.mean(torch.stack(max_list)))
        print("n_parameters",n_parameters)
        ## test
        test_l_segment,test_ssim,test_max_diff,test_mse=eval_loss(model, test_dl, segmentation_loss, True)
        

        # Collect Logger Info for testing evaluation
        logging.info(" ")
        logging.info('Training Complete...')
        logging.info('mean_val_ssim: {}'.format(torch.mean(torch.stack(val_ssim_list))))
        logging.info('mean_val_mse: {}'.format(torch.mean(torch.stack(mse_list))))
        logging.info('mean_val_max: {}'.format(torch.mean(torch.stack(max_list))))
        logging.info(" ")
        logging.info('Collecting Evaluation of Test Fold ...')
        logging.info('test_ssim: {}'.format(test_ssim))
        logging.info('test_max_diff: {}'.format(test_max_diff))
        logging.info('test_mse: {}'.format(test_mse))
        logging.info('Testing Complete.')


        return fold_train_loss,fold_valid_loss #epoch_loss_list,
    

if __name__ == '__main__':

    # Create logger with specific date/time 
    # log_folder = "E:\\Documents\\MRgLITT\\\saved_logs" # Change if needed
    log_folder = "E:\Documents\MRgLITT\data\\500J\segmentation\saved_logs"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"{log_folder}\\log_{current_time}.txt"
    logging.basicConfig(filename=log_filename, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    # MRI_path  = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/anatomicalProbesEye"
    # MRI_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\anatomicalProbesEye_no_t2"
    MRI_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\two_regions\segmentationData_2Regions_no_t2_3D"
    # MRI_path  = "E:\\Documents\\MRgLITT\\data\\2000J\\anatomicalProbesEye"
    # MRI_path  =  "/hpf/largeprojects/fkhalvati/Sara/course_project/histogram_equalized2"

    # thermometry_path = "/hpf/largeprojects/fkhalvati/Sara/course_project/temperatureData"
    # thermometry_path = "E:\\Documents\\MRgLITT\\data\\2000J\\temperatureData"
    thermometry_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\temperatureData_no_t2"
    # thermometry_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/preproccessed_data/masked"


    # our_dataset_train = Project_Dataloader_cv.MRIDataset(MRI_path , thermometry_path)
    # our_dataset_test= Project_Dataloader_cv.MRIDataset_test(our_dataset_train.get_dict()[1])

    splits=KFold(n_splits=9,shuffle=True,random_state=42)
    dataset=MRIDataset(None,MRI_path,thermometry_path,"id")
    args = make_parser().parse_args()
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)

    classes: int = 1
    n_segments = 2 # Number of segmentation classes
    test = False
    cuda = True
    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')
    if not test :  # training


        model = UNet()


        #________________________________________________base UNET_____________________________________________________
        # output_model_path = output_model_path = "E:\\Documents\\MRgLITT\\data\\2000J\\fixed_outputs"
        output_model_path = output_model_path = "E:\Documents\MRgLITT\data\\500J\original\preprocessed\\both_pre\\fixed_outputs"
        
        # "/hpf/largeprojects/fkhalvati/Sara/course_project/fixed_outputs/base_unet"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/eq_unet_test"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/masked_unet_test"
        #______________________________________________regression UNET__________________________________________________
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/base_regunet_test"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/eq_regunet_test"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/masked_regunet_test"
        #_______________________________________________attention UNET____________________________________________________
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/base_attunet_test"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/eq_attunet_test"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/code from AI course/fixed_outputs/masked_attunet_test"
        
        
        epoch_loss_list_ours = train_network(args, model, dataset,MRI_path,thermometry_path)

    best_score = 0.0
    best_model_name = ''
    model_dir =  output_model_path