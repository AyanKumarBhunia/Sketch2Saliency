import torch
import time
from salcap.dataloader.dataset import *
from salcap.model import *
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
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--eval_freq_iter', type=int, default=1)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--splitTrain', type=int, default=0.7)

    parser.add_argument('--numstart', default=-1, type=int)
    parser.add_argument('--numprint', default=200, type=int)
    parser.add_argument('--numval', default=2000, type=int)
    parser.add_argument('--numtrain', default=400000, type=int)
    parser.add_argument('--numworkers', default=8, type=int)
    parser.add_argument('--wr', default=5e-3, type=float)
    parser.add_argument('--wt', default=1e-2, type=float)

    hp = parser.parse_args()

    dataloader_Train_cap, vocab = get_dataloader(hp)
    dataloader_Train_sal, dataloader_Test_sal = get_dataloader_(hp)

    hp.vocab_size = len(dataloader_Train_cap.dataset.vocab)

    model = Caption(hp)
    model.to(device)
    step, best_accuracy = 0, 0

    for epoch in range(hp.max_epoch):
        for i_batch, (img, captions, lengths) in enumerate(dataloader_Train_cap):
            img, captions, lengths = img.to(device), captions.to(device), lengths.to(device)
            loss = model(img, captions, lengths)

            step += 1
            if (step + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, Loss Mod'
                      'el: {}, '
                      'Best Accuracy: {}'.format(epoch, i_batch, step, loss, best_accuracy))

            if (step + 1) % hp.eval_freq_iter == 0:

                with torch.no_grad():
                    maxfm, mae = model.val_sal(dataloader_Train_sal, epoch)
                    
