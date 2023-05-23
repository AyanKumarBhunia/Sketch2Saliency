import sys
import os
#sys.path.append(os.getcwd())
sys.path.append('/vol/research/ayanCV/CVPR2022/Sketch_Saliency-SketchX/')

import torch
import time
from salclsSketchy104.dataset import *
from salclsSketchy104.model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from Saliency_dataset import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Learning Deep Sketch Abstraction')


    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str, default=os.path.join(os.getcwd(), 'models'))
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--eval_freq_iter', type=int, default=10000)
    parser.add_argument('--print_freq_iter', type=int, default=100)
    parser.add_argument('--splitTrain', type=int, default=0.7)

    parser.add_argument('--numstart', default=-1, type=int)
    parser.add_argument('--numprint', default=200, type=int)
    parser.add_argument('--numval', default=2000, type=int)
    parser.add_argument('--numtrain', default=400000, type=int)
    parser.add_argument('--numworkers', default=8, type=int)
    parser.add_argument('--wr', default=5e-3, type=float)
    parser.add_argument('--wt', default=1e-2, type=float)

    hp = parser.parse_args()
    dataloader_Train_cls, dataloader_Test_cls = get_dataloader(hp)
    dataloader_Train_sal, dataloader_Test_sal = get_dataloader_(hp)

    model = Classification(hp)
    model.to(device)
    step, best_accuracy, maxfm, mae  = -1, 0, 0, 0

    for epoch in range(hp.max_epoch):
        for i_batch, (img, label) in enumerate(dataloader_Train_cls):
            img, label = img.to(device), label.to(device)
            loss = model(img,  label)

            
            if (step + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, Loss Mod'
                      'el: {}, '
                      'Best Accuracy: {}, maxfm {}, mae {}'.format(epoch, i_batch, step, loss, best_accuracy, maxfm, mae))

            if (step + 1) % hp.eval_freq_iter == 0:

                with torch.no_grad():
                    maxfm, mae = model.val_sal(dataloader_Train_sal, epoch)

            

            step += 1
                    
