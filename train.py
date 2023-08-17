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

import wandb

from collections import OrderedDict

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch Eye Gaze UNet')

    # Data
    parser.add_argument('--image_path', type=str, default="", help='image_path')
    parser.add_argument('--heatmaps_path', type=str, help='Heatmaps directory', default='')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    # parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

    # Training
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    # parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    # parser.add_argument('--lr', type=float, default=4e-3, help='initial learning rate')
    # parser.add_argument('--dropout', type=float, default=0, help='dropout')


    # Misc
    parser.add_argument('--gpus', type=str, default='0', help='Which gpus to use, -1 for CPU')
    parser.add_argument('--rseed', type=int, default=42, help='Seed for reproducibility')
    return parser


    

def train_network(config, args, model,test_ids, train_ids, valid_ids,image_path,thermometry_path): # data_ds change to test_ids and train_ids, valid_ids


    # Collect Logger Info for model parame6ters
    logging.info('Starting train_network()...')
    logging.info('Energy Level: 500J - Epochs: {} - Learning Rate: {} - Batch Size: {} - Extra: Augmentation(2region)'.format(config.epochs, config.lr, config.batch_size))
    # wandb.log({"Epochs": epoch, "lr": lr})

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = optim.Adam(model.parameters(), lr=0.004)

    segmentation_loss = nn.functional.l1_loss#nn.BCEWithLogitsLoss()# nn.CrossEntropyLoss()#GeneralizedDiceLoss()
    epoch_loss_list = []
    fold_train_loss=[]
    fold_valid_loss=[]
    val_ssim_list=[]
    n_parameters=[]
    max_list=[]
    mse_list=[]
    fold_ssim = []
    max_ssim=0

       

    # Create Test DataLoader
    # test_list=[data_ds[j] for j in test_idx]
    test_list= test_ids       
    test_ds=MRIDataset(test_list,image_path,thermometry_path,"image_heatmap")
    test_dl = DataLoader(test_ds)
    print("test idx", test_ds.id_list)

    # print("len test pngs", len(test_ds.id_list))

    development_ds = train_ids + valid_ids

    # print("dev",development_ds) 
    # print("len dev pngs", len(development_ds))
    
    
    for valid_fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(development_ds)))):
        print("train fold",train_idx)
        print("val fold",val_idx)

        
        train_list=[development_ds[j] for j in train_idx]
        valid_list=[development_ds[j] for j in val_idx]
        

        train_ds=MRIDataset(train_list,image_path,thermometry_path,"image_heatmap")
        valid_ds=MRIDataset(valid_list,image_path,thermometry_path,"image_heatmap")
 
        # result = valid_ds.get_dict()
        # val_list = [sublist[-1] for sublist in result] 
        # print(val_list)
        # print(valid_ds.id_list)

        train_dl =DataLoader(train_ds, batch_size=config.batch_size)  #in train_dl we have batch sized data  
        valid_dl = DataLoader(valid_ds, batch_size=config.batch_size)

        # print("len train pngs", len(train_ds.filenames))
        # print("len valid pngs", len(valid_ds.filenames))



        fold_train_loss.append([])
        fold_valid_loss.append([])
        fold_ssim.append([])
        print('Valid Fold {}'.format(valid_fold + 1))
        
        # # Transfer Learning Code
        # saved_model = "E:\\Documents\\MRgLITT\\data\\1000J\\fixed_outputs\\test_fold0_valid_fold8_Epoch_500.pth"
        # model = UNet()
        # state_dict = torch.load(saved_model)
        # model.load_state_dict(state_dict)

        # for param in model.parameters():
        #     param.requires_grad = False

        
        model=UNet()#regression_UNet()#UNet()#AttU_Net()#UNet()#AttU_Net()#UNet()
        #print("n_parameters",sum(p.numel() for p in model.parameters() if p.requires_grad))
        model=model.cuda()


        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        # optimizer = optim.Adam(model.parameters(), 0.004)
        optimizer.zero_grad()
        for epoch in range(config.epochs):
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
                print("epoch", epoch, "| validation SSIM:", ss.item(), "| val_segment:", val_segment.item())
                val_ssim_list.append(ss)
                mse_list.append(val_mse)
                max_list.append(val_max)

                #add fold ss to fold ssim list
                fold_ssim_list.append(ss)

                # if ss>=max_ssim:
                #     max_ssim=ss

                #     # deep copy save of current best model
                #     best_weights = deepcopy(model.state_dict())
                        
                #     #epoch_loss_list.append(epoch_loss)
                fold_train_loss[valid_fold].append(loss_segment.item())
                fold_valid_loss[valid_fold].append(val_segment.item())  
                fold_ssim[valid_fold].append(ss.item())
                wandb.log({'train loss per epoch':  loss_segment.item(), 'valid loss per epoch': val_segment.item()})

        # ('Energy Level: 500J - Epochs: {} - Learning Rate: {} - Batch Size: {} - Extra: '.format(config.epochs, config.lr, config.batch_size))

        if epoch== (config.epochs - 1):
            wandb.log({'mean l1 train loss in fold':  np.mean(fold_train_loss[valid_fold]), 'mean l1 valid loss in fold':  np.mean(fold_valid_loss[valid_fold]), 'mean ssim in fold:':np.mean(fold_ssim[valid_fold])})

            best_weights = deepcopy(model.state_dict())

            # save weight for best training fold based on validation loss
            for fold in range(2,valid_fold):
                if fold != valid_fold:
                    if np.mean(fold_valid_loss[valid_fold]) >= np.mean(fold_valid_loss[fold]):
            
                        # deep copy save of current best model
                        best_weights = deepcopy(model.state_dict())


            mean_fold_ssim = torch.mean(torch.stack(fold_ssim_list))

            #torch.save(model.state_dict(), output_model_path + f"/test_fold{test_fold}_valid_fold{valid_fold}_Epoch_{epoch+1}.pth")
            # print("train_loss_list:",fold_train_loss)
            # print("validation_loss_list",fold_valid_loss)

            logging.info(" ")
            logging.info('Add Evaluation for Valid Fold {}...'.format(valid_fold + 1))                           
            #mean_train_loss = torch.mean(torch.tensor(fold_train_loss))
            #mean_valid_loss = torch.mean(torch.tensor(fold_valid_loss))
            logging.info(f"mean_val_ssim: {mean_fold_ssim:.2f}/1")
            logging.info(f"mean val_l1: {np.mean(fold_train_loss[valid_fold]):.2f}/1")

                
    print("mean_val_ssim",torch.mean(torch.stack(val_ssim_list)))
    print("mean_val_mse",torch.mean(torch.stack(mse_list)))
    print("mean_val_max",torch.mean(torch.stack(max_list)))
    # print("n_parameters",n_parameters)
    ## test

    # Save
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_l_segment,test_ssim,test_max_diff,test_mse=eval_loss(model, test_dl, segmentation_loss, True)
    additional = "DataAug"
    torch.save(best_weights, output_model_path + f"/{config.epochs}_{config.lr}_{config.batch_size}_{test_ssim}_{additional}_.pth")


    wandb.log({'test_l_segment': test_l_segment.item(), 'test_ssim': test_ssim})
    
    # Collect Logger Info for testing evaluation
    logging.info(" ")
    logging.info('Training Complete...')        
    logging.info("mean_val_ssim: %.2f/1" % torch.mean(torch.stack(val_ssim_list)))
    logging.info('mean_val_max_diff: {:.2f}°C'.format(torch.mean(torch.stack(max_list))))
    logging.info('mean_val_mse: {:.2f}°C²'.format(torch.mean(torch.stack(mse_list))))
    logging.info(" ")
    logging.info('Collecting Evaluation of Test Fold ...')
    logging.info(f"test_ssim: {test_ssim:.2f}/1")
    logging.info('test_max_diff: {:.2f}°C'.format(test_max_diff))
    logging.info('test_mse: {:.2f}°C²'.format(test_mse))
    logging.info('Testing Complete.')

    wandb.log({'mean_val_ssim': torch.mean(torch.stack(val_ssim_list)), 'test_ssim': test_ssim})



    return fold_train_loss,fold_valid_loss #epoch_loss_list,
    
def main():

    wandb.init(project='HP-sweep')
    print(wandb.config)
    score = train_network(wandb.config,args, model, test_ids, train_ids, valid_ids,MRI_path,thermometry_path)
    wandb.log({'score': score})

def id_loader(unique_ids):

    seed_value = 42
    random.seed(seed_value)
    random.shuffle(unique_ids)

    # Split the list into 20% and 80% portions (test, train)
    split_index = int(len(unique_ids) * 0.2)
    test_ids = unique_ids[:split_index]
    train_valid = unique_ids[split_index:]

    # Split train/valid 
    split_index2 = int(len(train_valid) * 0.5)
    train_ids = train_valid[:split_index2]
    valid_ids = train_valid[split_index2:]

    print(len(test_ids), len(train_ids), len(valid_ids))
    print(test_ids, train_ids, valid_ids)

    return test_ids, train_ids, valid_ids




if __name__ == '__main__':

    # Create logger with specific date/time 
    log_folder = "E:\\Documents\\MRgLITT\\\saved_logs" # Change if needed
    # log_folder = "E:\\MRgLITT Data\\new data\\2021-12-21\\2_more_data_sub-folders_MTLE\\PNG\\saved_logs"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"{log_folder}\\log_{current_time}.txt"
    logging.basicConfig(filename=log_filename, format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    # MRI_path  = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/1500J/S2/anatomicalProbesEye"
    # MRI_path  = "E:\\Documents\\MRgLITT\\data\\500J\\first_acquisition\\anatomicalProbesEye"
   
    # MRI_path  = "E:\\Documents\\MRgLITT\\data\\500J\\original\\anatomicalProbesEye"
    # MRI_path  = "E:\Documents\MRgLITT\data\\500J\original\\anatomicalProbesEye"
    # MRI_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\anatomicalProbesEye_no_t2"
    # MRI_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\two_regions\segmentationData_2Regions_1D_Augmented"
    MRI_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\two_regions\segmentationData_2Regions_1D_Augmented"
    # MRI_path  =  "/hpf/largeprojects/fkhalvati/Sara/course_project/histogram_equalized2"

    # thermometry_path = "/hpf/largeprojects/fkhalvati/Sara/course_project/temperatureData"
    # thermometry_path = "E:\\Documents\\MRgLITT\\data\\500J\\original\\temperatureData"
    # thermometry_path = "E:\Documents\MRgLITT\data\\500J\original\preprocessed\masked"
    # thermometry_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\temperatureData_no_t2"
    thermometry_path = "E:\Documents\MRgLITT\data\\500J\segmentation\\two_regions\\temperatureData_no_t2_Augmented"
    # thermometry_path = "E:/MRgLITT Data/new data/2021-12-21/2
    # 
    # _more_data_sub-folders_MTLE/PNG/1500J/S2/temperatureData"
    # thermometry_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/500J/S2/preproccessed_data/masked"

    # segmentation_path = "E:\\Documents\\MRgLITT\\data\\500J\\original\\segmentationData_Corrected"


    # our_dataset_train = Project_Dataloader_cv.MRIDataset(MRI_path , thermometry_path)
    # our_dataset_test= Project_Dataloader_cv.MRIDataset_test(our_dataset_train.get_dict()[1])

    splits=KFold(n_splits=9,shuffle=True,random_state=42)

    #get test ids, get train ids, (new function)
    # 



    dataset=MRIDataset(None,MRI_path,thermometry_path,"image_heatmap") # image_heatmap
    
    # items = os.listdir(MRI_path)
    # png_count = len(items)
    # num_list = list(range(png_count))

    your_dict = dataset.filenames
    first_items = {key: value[0] for key, value in your_dict.items()}
    unique_ids = list(OrderedDict.fromkeys(first_items.values()))
    print(len(unique_ids))


    test_ids, train_ids, valid_ids = id_loader(unique_ids)

    # print(test_ids)

    # print(train_ids)
    # print(valid_ids)
    # dev = train_ids+valid_ids
    # # print(len(dev))
    # print(dev)
    

    # print("test:",test_ids)
    # print()
    # print("train:",train_ids)
    # print()
    # print("valid:",valid_ids)
    # print()
    # print("train/valid", train_ids,valid_ids)

    # train_valid = train_ids.extend(valid_ids)
    # print(train_valid)

    args = make_parser().parse_args()
    random.seed(args.rseed)
    np.random.seed(args.rseed)
    torch.manual_seed(args.rseed)

    classes: int = 1
    n_segments = 2 # Number of segmentation classes
    test = False
    cuda = True
    args.device = torch.device("cuda:"+ args.gpus) if cuda else torch.device('cpu')

    sweep_configuration = {
        'method': 'bayes',
        'name': 'sweep',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_segment'
            },
        'parameters': {
            'batch_size': {'values': [16]},
            'epochs': {'values': [200]},
            # 'epochs': {'values': [5]},
            # 'lr': {'max': 0.003, 'min': 0.0005}
            'lr':  { 'values': [0.0022]}
        }
    }

    sweep_id = wandb.sweep(
        sweep = sweep_configuration,
        project = "HP-sweep"

    )

    if not test :  # training

        model = UNet()

        # saved_model = "E:\\Documents\\MRgLITT\\data\\500J\\fixed_outputs\\test_fold0_valid_fold8_Epoch_500.pth"

        # state_dict = torch.load(saved_model)
        # model.load_state_dict(state_dict)
        # model.finalconv.requires_grad = False   


        #________________________________________________base UNET_____________________________________________________
        output_model_path = output_model_path = "E:\Documents\MRgLITT\data\\500J\original\preprocessed\\both_pre\\fixed_outputs"
        # output_model_path = "E:/MRgLITT Data/new data/2021-12-21/2_more_data_sub-folders_MTLE/PNG/1500J/S2/output"
        
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
        
        
        # epoch_loss_list_ours = train_network(args, model, dataset,MRI_path,thermometry_path)
        wandb.agent(sweep_id, function=main, count=2)

        # score = train_network(wandb.config,model,dataset,MRI_path,thermometry_path)
        # wandb.log({'score': score})

    best_score = 0.0
    best_model_name = ''
    model_dir =  output_model_path