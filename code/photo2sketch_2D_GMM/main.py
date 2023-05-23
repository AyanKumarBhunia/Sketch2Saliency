import sys
import os
sys.path.append(os.getcwd())
sys.path.append('/vol/research/ayanCV/CVPR2022/Sketch_Saliency-SketchX/')
sys.path.append('/home/media/CVPR_2022/Sketch_Saliency-SketchX/')

import torch
from model import Photo2Sketch
from dataset import get_dataloader, get_dataloaderShoeV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import sys
import os

from rasterize import batch_rasterize_relative
from torchvision.utils import save_image
import time
import os
import numpy as np
# device = 'cpu'
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Photo2Sketch')
    parser.add_argument('--dataset_name', type=str, default='sketchy')
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str,
                        default=os.path.join(os.getcwd(), 'models'))
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)

    # parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)


    parser.add_argument('--enc_rnn_size', default=256)
    parser.add_argument('--dec_rnn_size', default=512)
    parser.add_argument('--z_size', default=128)

    parser.add_argument('--num_mixture', default=20)
    parser.add_argument('--input_dropout_prob', default=0.9)
    parser.add_argument('--output_dropout_prob', default=0.9)
    parser.add_argument('--batch_size_sketch_rnn', default=100)

    parser.add_argument('--kl_weight_start', default=0.01)
    parser.add_argument('--kl_decay_rate', default=0.99995)
    parser.add_argument('--kl_tolerance', default=0.2)
    parser.add_argument('--kl_weight', default=1.0)

    parser.add_argument('--learning_rate', default=0.0001)
    parser.add_argument('--decay_rate', default=0.9999)
    parser.add_argument('--min_learning_rate', default=0.00001)
    parser.add_argument('--grad_clip', default=1.)


    # parser.add_argument('--sketch_rnn_max_seq_len', default=200)

    hp = parser.parse_args()

    print(hp)
    model = Photo2Sketch(hp)
    model.to(device)
    # model.load_state_dict(torch.load('/home/media/CVPR_2022/Sketch_Saliency-SketchX/models/Photo2Sketch_2D_50000.pth'))
    # model.requires_grad_(False)


    """ Model Training Image2Sketch """
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    # dataloader_Train, dataloader_Test = get_dataloaderShoeV2(hp)
    step = 0
    loss_best = 0

 

    # for batch_data in dataloader_Train:
    #     print(batch_data['positive_img'].shape)
    #     print(batch_data['sketch_img'].shape)

    #     save_image(batch_data['sketch_img'], 'sketch_img.jpg')
    #     save_image(batch_data['positive_img'], 'positive_img.jpg')

    #     print(batch_data['absolute_fivepoint'].shape)
    #     print(batch_data['relative_fivepoint'].shape)

    #     print(batch_data['seq_len'])
    #     save_image(1. - batch_rasterize_relative(batch_data['relative_fivepoint']), 'relative_fivepoint.jpg')
    #     break

    print(len(dataloader_Train))
    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            rgb_image = batch_data['positive_img'].to(device)
            sketch_vector = batch_data['relative_fivepoint'].to(device).permute(1, 0, 2).float() # Seq_Len, Batch, Feature

            length_sketch = batch_data['seq_len'].to(device) -1 #TODO: Relative coord has one less
            sketch_name = batch_data['sketch_path']

            hp.max_seq_len = batch_data['max_seq_len']
            sup_p2s_loss, kl_cost_rgb, total_loss = \
                    model.Image2Sketch_Train(rgb_image, sketch_vector, length_sketch, step, sketch_name)
            
            step += 1

            if step % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(hp.saved_models,  'Photo2Sketch_2D_' + str(step) + '.pth'))