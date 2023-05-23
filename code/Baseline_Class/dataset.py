import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import cv2
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy

#  1. Sketchy normal split.
#  RGB Images, photoMaps, Sketchy

unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon',
                  'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree',
                  'wheelchair', 'windmill', 'window']
from os import listdir
from os.path import isfile, join


class Sketchy_Dataset(data.Dataset):
    def __init__(self, hp, mode = 'Train'):

        self.hp = hp
        self.mode = mode
        #self.training = copy.deepcopy(hp.training)

        self.photo_path = os.path.join(hp.base_dir, 'Dataset/Sketchy/Extended_Photo')


        self.photo_list = [os.path.join(root, file) for f in listdir(self.photo_path) if f not in unseen_classes for root, dirs, files in os.walk(join(self.photo_path , f)) for file in files if file.endswith('.jpg')]
        self.photo_classes = [f for f in listdir(self.photo_path) if f not in unseen_classes]
        self.classes = self.photo_classes + unseen_classes
        self.photo_test_list = random.sample(self.photo_list, int(0.2*(len(self.photo_list))))
        self.photo_train_list = list(set(self.photo_list) - set(self.photo_test_list))


        self.total_class = len(self.classes)
        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(self.classes):
            self.num2name[num] = val
            self.name2num[val] = num

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

        print('Total Training Sample {}'.format(len(self.photo_train_list)))
        print('Total Training Testing Sample {}'.format(len(self.photo_test_list)))


    def __getitem__(self, item):

        if self.mode == 'Train':

            photo = Image.open(self.photo_train_list[item]).convert('RGB')
            class_name = self.photo_train_list[item].split('/')[-2]

            n_flip = random.random()
            if n_flip > 0.5:
                photo = F.hflip(photo)

            photo = self.train_transform(photo)
            label = self.name2num[class_name]
            
            return  photo, label

        elif self.mode == 'Test':

            photo = Image.open(self.photo_test_list[item]).convert('RGB')
            class_name = self.photo_test_list[item].split('/')[-2]

            n_flip = random.random()
            if n_flip > 0.5:
                photo = F.hflip(photo)

            photo = self.test_transform(photo)
            label = self.name2num[class_name]

            return photo, label

    def __len__(self):
        if self.mode == 'Train':
            return len(self.photo_train_list)
        elif self.mode == 'Test':
            return len(self.photo_test_list)




def get_transform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize((256, 256))])
    elif type is 'Test':
        transform_list.extend([transforms.Resize((256, 256))])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


def get_dataloader(hp):

    dataset_Train  = Sketchy_Dataset(hp, mode = 'Train')
    # hp.training =  'sketch'
    dataset_Test = Sketchy_Dataset(hp, mode='Test')
    

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))

    dataset_Train_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))


    return dataloader_Train, dataset_Test