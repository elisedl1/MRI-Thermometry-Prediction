import os
import re
import glob
import torch
import random
import numpy as np
import pandas as pd
# import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
# import cv2
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import argparse
from torchvision.io import read_image
import torchvision.transforms as T


class MRIDataset2(Dataset):
    """To create the dataset with patient uniq ID"""

    def __init__(self,id_list,image_path,thermometry_path,input_type):
        self.image_path = image_path
        self.thermometry_path = thermometry_path
        self.images=[]
        self.heatmaps=[]
        self.filenames=[]
        self.id_dict={}
        self.id_list=id_list
        self.input_type=input_type
        # print("input type",self.input_type)
        #print(os.listdir(self.image_path))
        for (dirpath, dirnames, filenames) in os.walk(self.image_path):
            #for filename in sorted(filenames,key="natural_keys"):
            for filename in sorted(filenames):
               # print("filenames",filenames)
                
                #print("MRI filename: ", filename)
                filepath = os.path.join(dirpath, filename)
                #self.images.append(torch.tensor(cv2.imread(filepath)))
               # print("gggggg",filepath)
                if "png" in filepath:
                    self.images.append(read_image(filepath))
                    self.filenames.append(filename[3:8])
                #    print(len(self.images))
               #     print(len(self.filename))

        for (dirpath, dirnames, filenames) in os.walk(self.thermometry_path):
            #for filename in sorted(filenames,key="natural_keys"):
            for filename in sorted(filenames):
                #print("Thermometry filename: ", filename)
                filepath = os.path.join(dirpath, filename)
                #self.heatmaps.append(cv2.imread(filepath).ToTensor())
                self.heatmaps.append(read_image(filepath))
    def __len__(self):
        if self.input_type=="id":
            return len(self.train_test_id_provider())
        elif self.input_type=="image_heatmap":
            return len(self.get_dict())
        
    def train_test_id_provider(self):
        for i in range(len(os.listdir(self.image_path))):
            self.id_dict[self.filenames[i]]=[]
        for i in range(len(os.listdir(self.image_path))):
            self.id_dict[self.filenames[i]].append([self.images[i],self.heatmaps[i]])
        uniq_id=list(set(self.filenames))
        #print("uniq ID len", len(uniq_id)) 
        random.seed(42)
        random.shuffle(uniq_id)
        return uniq_id
    
    def get_dict(self):
        #make a dictionary from whole dataset. 
        #LP-001-01, LP001-02, LP001-03, LP002-01, LP002-08 -> {LP-001:[[heatmap1, MRI1], [heatmap2, MRI2], [heatmap3, MRI3], LP-002:[[heatmap1,MRI1], [ heatmap2, MRI2]]}
        for i in range(len(os.listdir(self.image_path))): 
            self.id_dict[self.filenames[i]]=[]
        for i in range(len(os.listdir(self.image_path))): 
            self.id_dict[self.filenames[i]].append([self.images[i],self.heatmaps[i]]) 
        
        images_heatmap_ls=[]
        #find the uniq ids from the complete dictionary of all patient ids with all sub folders (e.g. LP001-01, LP001-02)
        for i in self.id_list:
            for j in self.id_dict[i]:
                images_heatmap_ls.append(j)
        # print("image_heatmap_ls",images_heatmap_ls[0])
        # print("image_heatmap_ls",len(images_heatmap_ls))
        return images_heatmap_ls
    
    def __getitem__(self, idx):
        
        if self.input_type=="id":
        #output of the class
        
            unique_ids=self.train_test_id_provider()
            unique_id=unique_ids[idx]
            
            return unique_id
        elif self.input_type=="image_heatmap":
            # print(self.get_dict()[idx][0].shape)
            
            image = self.get_dict()[idx][0][0,:,:].reshape(1,51,51)
            heatmap=self.get_dict()[idx][1][0,:,:].reshape(1,51,51)

                        
            return image, heatmap 
        


class MRIDataset(Dataset):
    """To create the dataset with patient uniq ID"""

    def __init__(self,id_list,image_path,thermometry_path,input_type):
        super(MRIDataset,self).__init__()

        self.image_path = image_path
        self.thermometry_path = thermometry_path
        self.images=[]
        self.heatmaps=[]
        self.filenames={}
        self.id_dict={}
        self.id_list=id_list
        self.input_type=input_type
        # print("input type",self.input_type)
        #print(os.listdir(self.image_path))
        for (dirpath, dirnames, filenames) in os.walk(self.image_path):
            #for filename in sorted(filenames,key="natural_keys"):
            for file_num,filename in enumerate(sorted(filenames)):
               # print("filenames",filenames)
                
                #print("MRI filename: ", filename)
                filepath = os.path.join(dirpath, filename)
                #self.images.append(torch.tensor(cv2.imread(filepath)))
               # print("gggggg",filepath)
                if "png" in filepath:
                    
                    self.images.append(read_image(filepath))
                    
                    self.filenames[file_num]=[filename[3:8],filename] ##

                #    print(len(self.images))
               #     print(len(self.filename))

        for (dirpath, dirnames, filenames) in os.walk(self.thermometry_path):
            #for filename in sorted(filenames,key="natural_keys"):
            for filename in sorted(filenames):
                #print("Thermometry filename: ", filename)
                filepath = os.path.join(dirpath, filename)
                #self.heatmaps.append(cv2.imread(filepath).ToTensor())
                self.heatmaps.append(read_image(filepath))

    def __len__(self):
        if self.input_type=="id":
            return len(self.train_test_id_provider())
        elif self.input_type=="image_heatmap":
            return len(self.get_dict())
        
    def train_test_id_provider(self):
        for i in range(len(os.listdir(self.image_path))):
            self.id_dict[self.filenames[i][0]]=[]
        for i in range(len(os.listdir(self.image_path))):
            self.id_dict[self.filenames[i][0]].append([self.images[i],self.heatmaps[i]])
        uniq_id=list(set([self.filenames[num_uniq_ids][0] for num_uniq_ids in list(self.filenames.keys())]))
        
        #print("uniq ID len", len(uniq_id))
        # random.seed(42)
        # random.shuffle(uniq_id)
        return uniq_id
    
    def get_dict(self):
        #make a dictionary from whole dataset. 
        #LP-001-01, LP001-02, LP001-03, LP002-01, LP002-08 -> {LP-001:[[heatmap1, MRI1], [heatmap2, MRI2], [heatmap3, MRI3], LP-002:[[heatmap1,MRI1], [ heatmap2, MRI2]]}
        for i in range(len(os.listdir(self.image_path))): 
            self.id_dict[self.filenames[i][0]]=[]
        for i in range(len(os.listdir(self.image_path))): 
            self.id_dict[self.filenames[i][0]].append([self.images[i],self.heatmaps[i],self.filenames[i][1]]) 
        
        images_heatmap_ls=[]
        #find the uniq ids from the complete dictionary of all patient ids with all sub folders (e.g. LP001-01, LP001-02)
        for i in self.id_list:
            for j in self.id_dict[i]:
                images_heatmap_ls.append(j)
        # print("image_heatmap_ls",images_heatmap_ls[0])
        # print("image_heatmap_ls",len(images_heatmap_ls))
        return images_heatmap_ls
    
    def __getitem__(self, idx):

        unique_ids=self.train_test_id_provider()
        if self.input_type=="id":
        #output of the class
        
            
            unique_id=unique_ids[idx]
            
            return unique_id
        elif self.input_type=="image_heatmap":
            # print(self.get_dict()[idx][0].shape)
            
            image = self.get_dict()[idx][0][0,:,:].reshape(1,51,51)
            heatmap=self.get_dict()[idx][1][0,:,:].reshape(1,51,51)
            file_name=self.get_dict()[idx][2]
                  
            return image, heatmap,file_name 


# import os
# import re
# import glob
# import torch
# import random
# import numpy as np
# import pandas as pd
# # import SimpleITK as sitk
# from PIL import Image
# from torch.utils.data import Dataset
# # import cv2
# from torch.utils.data import DataLoader
# import argparse
# import matplotlib.pyplot as plt
# import argparse
# from torchvision.io import read_image
# import torchvision.transforms as T

# class MRIDataset_id(Dataset):
#     """To create the dataset with patient uniq ID"""

#     def __init__(self,image_path,thermometry_path):
#         self.image_path = image_path
#         self.thermometry_path = thermometry_path
#         self.images=[]
#         self.heatmaps=[]
#         self.filenames=[]
#         self.id_dict={}


#         #print(os.listdir(self.image_path))
#         for (dirpath, dirnames, filenames) in os.walk(self.image_path):
#             #for filename in sorted(filenames,key="natural_keys"):
#             for filename in sorted(filenames):
#                # print("filenames",filenames)
                
#                 #print("MRI filename: ", filename)
#                 filepath = os.path.join(dirpath, filename)
#                 #self.images.append(torch.tensor(cv2.imread(filepath)))
#                # print("gggggg",filepath)
#                 if "png" in filepath:
#                     self.images.append(read_image(filepath))
#                     self.filenames.append(filename[3:8])
#                 #    print(len(self.images))
#                #     print(len(self.filename))

#         for (dirpath, dirnames, filenames) in os.walk(self.thermometry_path):
#             #for filename in sorted(filenames,key="natural_keys"):
#             for filename in sorted(filenames):
#                 #print("Thermometry filename: ", filename)
#                 filepath = os.path.join(dirpath, filename)
#                 #self.heatmaps.append(cv2.imread(filepath).ToTensor())
#                 self.heatmaps.append(read_image(filepath))
#     def __len__(self):
#         return len(self.train_test_id_provider())
    
#     def train_test_id_provider(self):
#         for i in range(319):
#             self.id_dict[self.filenames[i]]=[]
#         for i in range(319):
#             self.id_dict[self.filenames[i]].append([self.images[i],self.heatmaps[i]])
#         uniq_id=list(set(self.filenames))
#         #print("uniq ID len", len(uniq_id))
#         random.seed(42)
#         random.shuffle(uniq_id)
#         return uniq_id
    
#     def __getitem__(self, idx):
#         #output of the class
#         unique_ids=self.train_test_id_provider()
#         unique_id=unique_ids[idx]
#         return unique_id


# class MRIDataset_im(Dataset):
#     """To create the dataset with thermometry and MRI images with uniq ID"""

#     def __init__(self,id_list,image_path,thermometry_path):
#         self.image_path = image_path
#         self.thermometry_path = thermometry_path
#         self.id_list=id_list #uniq id list = 81 in this step
#         self.images=[]
#         self.heatmaps=[]
#         self.filenames=[]
#         self.id_dict={}

#         for (dirpath, dirnames, filenames) in os.walk(self.image_path):
#             #for filename in sorted(filenames,key="natural_keys"):
#             for filename in sorted(filenames):
#                # print("filenames",filenames)
                
#                 #print("MRI filename: ", filename)
#                 filepath = os.path.join(dirpath, filename)
#                 #self.images.append(torch.tensor(cv2.imread(filepath)))
#                # print("gggggg",filepath)
#                 if "png" in filepath:
#                     self.images.append(read_image(filepath))
#                     self.filenames.append(filename[3:8])
#                 #    print(len(self.images))
#                #     print(len(self.filename))

#         for (dirpath, dirnames, filenames) in os.walk(self.thermometry_path):
#             #for filename in sorted(filenames,key="natural_keys"):
#             for filename in sorted(filenames):
#                 #print("Thermometry filename: ", filename)
#                 filepath = os.path.join(dirpath, filename)
#                 #self.heatmaps.append(cv2.imread(filepath).ToTensor())
#                 self.heatmaps.append(read_image(filepath))
#     def __len__(self):
#         return len(self.get_dict())

#     def get_dict(self):
#         #make a dictionary from whole dataset. 
#         #LP-001-01, LP001-02, LP001-03, LP002-01, LP002-08 -> {LP-001:[[heatmap1, MRI1], [heatmap2, MRI2], [heatmap3, MRI3], LP-002:[[heatmap1,MRI1], [ heatmap2, MRI2]]}
#         for i in range(319):
#             self.id_dict[self.filenames[i]]=[]
#         for i in range(319):
#             self.id_dict[self.filenames[i]].append([self.images[i],self.heatmaps[i]]) 
        
#         images_heatmap_ls=[]
#         #find the uniq ids from the complete dictionary of all patient ids with all sub folders (e.g. LP001-01, LP001-02)
#         for i in self.id_list:
#             for j in self.id_dict[i]:
#                 images_heatmap_ls.append(j)
#         return images_heatmap_ls

#     def __getitem__(self, idx):
#         image = self.get_dict()[idx][0][0,:,:].reshape(1,51,51)
#         heatmap=self.get_dict()[idx][1][0,:,:].reshape(1,51,51)
#         return image, heatmap

