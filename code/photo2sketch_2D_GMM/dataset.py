import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random

from rasterize import rasterize_Sketch
from utils import *
from torchvision.utils import save_image
import torchvision.transforms.functional as F 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon',
                  'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree',
                  'wheelchair', 'windmill', 'window']



class Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.base_dir, 'Dataset', hp.dataset_name, hp.dataset_name + '_all.pickle') # _all.pickle
        self.root_dir = os.path.join(hp.base_dir, 'Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            train_sketch, test_sketch, _, _ = pickle.load(fp)
        
        coordinate_path = os.path.join(hp.base_dir, 'Dataset', hp.dataset_name, hp.dataset_name + '_RDP_6') # _all.pickle
        with open(coordinate_path, 'rb') as fp:
           self.Coordinate = pickle.load(fp)

        train_set = [x for x in train_sketch if x.split('/')[0] not in unseen_classes]
        train1_set = [x for x in test_sketch if x.split('/')[0] not in unseen_classes]
        self.Train_Sketch = train_set + train1_set

        Test_Sketch = [x for x in train_sketch if x.split('/')[0] in unseen_classes]
        Test1_Sketch = [x for x in test_sketch if x.split('/')[0] in unseen_classes]

        self.Test_Sketch = Test_Sketch + Test1_Sketch

        self.classes = list(set([x.split('/')[0] for x in self.Train_Sketch]))
        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(self.classes):
            self.num2name[num] = val
            self.name2num[val] = num

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')


        # """" Preprocess offset coordinates """
        # self.Coordinate = {}
        # for key in Coordinate.keys():
        #     if len(Coordinate[key]) < 2001:
        #         self.Coordinate[key] = Coordinate[key]

        len_x = []
        for key in self.Coordinate.keys():
            len_x.append(len(self.Coordinate[key]))
        
        print(np.sum(np.array(len_x)>200))
        # """" Preprocess offset coordinates """
        # self.Offset_Coordinate = {}
        # for key in self.Coordinate.keys():
        #     self.Offset_Coordinate[key] = to_delXY(self.Coordinate[key])
        # data = []
        # for sample in self.Offset_Coordinate.values():
        #     data.extend(sample[:, 0])
        #     data.extend(sample[:, 1])
        # data = np.array(data)
        # self.scale_factor = np.std(data)

        # for key in self.Coordinate.keys():
        #     self.Offset_Coordinate[key][:, :2] /= self.scale_factor

        # """" <<< Preprocess offset coordinates >>> """
        # """" <<<           Done                >>> """

        self.scale_factor = 54.48767345581628 # 54.48767345581628 vs 11.659693663551407

    def __getitem__(self, item):

        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = sketch_path.split('/')[0] + '/' + sketch_path.split('/')[1].split('-')[0]
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.jpg')

            vector_x = self.Coordinate[sketch_path]

            if len(vector_x) > 201:
                return self[item + 1]

            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            img = Image.open(positive_path).convert('RGB')

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                img = F.hflip(img)
                sketch_points[:, 0] = -sketch_points[:, 0] + 256.

            sketch_img = self.train_transform(sketch_img)
            img = self.train_transform(img)

            #-----------------------------Added----------------------------------------#
            sketch_delta =  to_delXY(sketch_points)
            sketch_delta[:, :2] /= self.scale_factor

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': img, 'positive_path': positive_sample,
                      'absolute_points': sketch_points, 'label': self.name2num[sketch_path.split('/')[0]],
                      'relative_points': sketch_delta
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img, sketch_points = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB'))

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            sketch_delta =  to_delXY(sketch_points)
            sketch_delta[:, :2] /= self.scale_factor

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'absolute_points': sketch_points, 'relative_points':sketch_delta}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


def get_dataloader(hp):

    dataset_Train = Dataset(hp, mode='Train')
    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                       num_workers=int(hp.nThreads), collate_fn=collate_self_Train)

    dataset_Test = Dataset(hp, mode='Test')
    dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False, collate_fn=collate_self_Test,
                                      num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test


def get_transform(type):
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type == 'Test':
        transform_list.extend([transforms.Resize(256)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)



def collate_self_Train(batch):
    batch_mod = {'sketch_img': [], 'sketch_path': [],
                 'positive_img': [], 'positive_path': [],
                 'absolute_fivepoint': [], 'seq_len': [], 'label': [],
                 'relative_fivepoint': []
                 }

    max_len = max([len(x['absolute_points']) for x in batch])
    batch_mod['max_seq_len'] = max_len + 1

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
 
        five_point, len_seq = to_Five_Point(i_batch['absolute_points'], max_len)
        # First time step is [0, 0, 0, 0, 0] as start token.
        batch_mod['absolute_fivepoint'].append(torch.tensor(five_point))
        batch_mod['seq_len'].append(len_seq)
        batch_mod['label'].append(i_batch['label'])
        #--------------------Added-----------------------------#
        five_point_rela, len_seq = to_Five_Point(i_batch['relative_points'], max_len)
        batch_mod['relative_fivepoint'].append(torch.tensor(five_point_rela))
        #---------------------------------------------------------#

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
 
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])
    batch_mod['label'] = torch.tensor(batch_mod['label'])

    #-------------------------------Added-----------------------#
    batch_mod['relative_fivepoint'] = torch.stack(batch_mod['relative_fivepoint'], dim=0)
    batch_mod['absolute_fivepoint'] = torch.stack(batch_mod['absolute_fivepoint'], dim=0)
    #--------------------------------------------------------#

    return batch_mod


def collate_self_Test(batch):
    batch_mod = {'sketch_img': [], 'sketch_path': [],
                 'positive_img': [], 'positive_path': [],
                 'absolute_fivepoint': [], 'seq_len': [],
                 'relative_fivepoint': []
                 }


    max_len = max([len(x['sketch_points']) for x in batch])

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])

        five_point, len_seq = to_Five_Point(i_batch['absolute_points'], max_len)
        # First time step is [0, 0, 0, 0, 0] as start token.
        batch_mod['absolute_fivepoint'].append(torch.tensor(five_point))
        batch_mod['seq_len'].append(len_seq)

        #--------------------Added-----------------------------#
        five_point_rela, len_seq = to_Five_Point(batch_mod['relative_points'], max_len)
        batch_mod['relative_fivepoint'].append(torch.tensor(five_point_rela))
        #---------------------------------------------------------#

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    batch_mod['sketch_five_point'] = torch.stack(batch_mod['sketch_five_point'], dim=0)
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])

    #-------------------------------Added-----------------------#
    batch_mod['absolute_fivepoint'] = torch.stack(batch_mod['relative_fivepoint'], dim=0)
    batch_mod['relative_fivepoint'] = torch.stack(batch_mod['absolute_fivepoint'], dim=0)
    #--------------------------------------------------------#

    return batch_mod



class Photo2Sketch_Dataset(data.Dataset):

    def __init__(self, hp, mode):
        super(Photo2Sketch_Dataset, self).__init__()



        coordinate_path = os.path.join(hp.base_dir, 'Dataset', 'ShoeV2', 'ShoeV2_RDP_3')
        self.root_dir = os.path.join(hp.base_dir, 'Dataset', 'ShoeV2')
 


        self.hp = hp
        self.mode = mode
        # hp.root_dir = '/home/media/On_the_Fly/Code_ALL/Final_Dataset'
        hp.dataset_name = 'ShoeV2'
        hp.seq_len_threshold = 251
 
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)


        seq_len_threshold = 81
        coordinate_refine = {}
        seq_len = []
        for key in self.Coordinate.keys():
            if len(self.Coordinate[key]) < seq_len_threshold:
                coordinate_refine[key] = self.Coordinate[key]
                seq_len.append(len(self.Coordinate[key]))
        self.Coordinate = coordinate_refine
        hp.max_seq_len = max(seq_len)
        hp.average_seq_len = int(np.round(np.mean(seq_len) + 0.5*np.std(seq_len)))

        # greater_than_average = 0
        # for seq in seq_len:
        #     if seq > self.hp.average_len:
        #         greater_than_average +=1

        self.Train_Sketch = [x for x in self.Coordinate if ('train' in x) and (len(self.Coordinate[x]) < seq_len_threshold)]  # separating trains
        self.Test_Sketch = [x for x in self.Coordinate if ('test' in x) and (len(self.Coordinate[x]) < seq_len_threshold)]    # separating tests

        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

        # # seq_len = []
        # # for key in self.Coordinate.keys():
        # #     seq_len += [len(self.Coordinate[key])]
        # # plt.hist(seq_len)
        # # plt.savefig('histogram of number of Coordinate Points.png')
        # # plt.close()
        # # hp.max_seq_len = max(seq_len)
        # hp.max_seq_len = 130


        """" Preprocess offset coordinates """
        self.Offset_Coordinate = {}
        for key in self.Coordinate.keys():
            self.Offset_Coordinate[key] = to_delXY(self.Coordinate[key])
        data = []
        for sample in self.Offset_Coordinate.values():
            data.extend(sample[:, 0])
            data.extend(sample[:, 1])
        data = np.array(data)
        scale_factor = np.std(data)

        for key in self.Coordinate.keys():
            self.Offset_Coordinate[key][:, :2] /= scale_factor

        """" <<< Preprocess offset coordinates >>> """
        """" <<<           Done                >>> """



    def __getitem__(self, item):

        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = Image.open(positive_path).convert('RGB')

            sketch_abs = self.Coordinate[sketch_path]
            sketch_delta = self.Offset_Coordinate[sketch_path]

            sketch_img, _ = rasterize_Sketch(sketch_abs, side_norm=256)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            # sketch_img = TF.hflip(sketch_img)
            # positive_img = TF.hflip(positive_img)
            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)

            #################################### #################################### ####################################

            absolute_coordinate = np.zeros((self.hp.max_seq_len, 3))
            relative_coordinate = np.zeros((self.hp.max_seq_len, 3))
            absolute_coordinate[:sketch_abs.shape[0], :] = sketch_abs
            relative_coordinate[:sketch_delta.shape[0], :] = sketch_delta
            #################################### #################################### ####################################

            # sample = {'sketch_img': sketch_img,
            #           'sketch_path': sketch_path,
            #           'absolute_coordinate':absolute_coordinate,
            #           'relative_coordinate': relative_coordinate,
            #           'sketch_length': int(len(sketch_abs)),
            #           'absolute_fivePoint': to_FivePoint(sketch_abs, self.hp.max_seq_len),
            #           'relative_fivePoint': to_FivePoint(sketch_delta, self.hp.max_seq_len),
            #           'positive_img': positive_img,
            #           'positive_path': positive_sample}

            sample = {'sketch_path': sketch_path, 'seq_len': int(len(sketch_abs)),
                        'relative_fivepoint': to_Five_Point(sketch_delta, self.hp.max_seq_len)[0][1:,:],
                        'positive_img': positive_img}


        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            sketch_abs = self.Coordinate[sketch_path]
            sketch_delta = self.Offset_Coordinate[sketch_path]

            sketch_img = rasterize_Sketch(sketch_abs)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            sketch_img = self.test_transform(sketch_img)
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            #################################### #################################### ####################################

            absolute_coordinate = np.zeros((self.hp.max_seq_len, 3))
            relative_coordinate = np.zeros((self.hp.max_seq_len, 3))
            absolute_coordinate[:sketch_abs.shape[0], :] = sketch_abs
            relative_coordinate[:sketch_delta.shape[0], :] = sketch_delta
            #################################### #################################### ####################################

            # sample = {'sketch_img': sketch_img,
            #           'sketch_path': sketch_path,
            #           'absolute_coordinate':absolute_coordinate,
            #           'relative_coordinate': relative_coordinate,
            #           'sketch_length': int(len(sketch_abs)),
            #           'absolute_fivePoint': to_FivePoint(sketch_abs, self.hp.max_seq_len),
            #           'relative_fivePoint': to_FivePoint(sketch_delta, self.hp.max_seq_len),
            #           'positive_img': positive_img,
            #           'positive_path': positive_sample}

            sample = { 'sketch_path': sketch_path,
                        'length': int(len(sketch_abs)),
                        'relative_fivepoint': to_Five_Point(sketch_delta, self.hp.max_seq_len)[0][1:,:],
                        'photo': positive_img}

        return sample
    
    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


def get_dataloaderShoeV2(hp):

    dataset_Train  = Photo2Sketch_Dataset(hp, mode = 'Train')


    dataset_Test  = Photo2Sketch_Dataset(hp, mode = 'Test')

    # dataset_Train = torch.utils.data.ConcatDataset([dataset_Train, dataset_Test])

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads))

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads))

    return dataloader_Train, dataloader_Test