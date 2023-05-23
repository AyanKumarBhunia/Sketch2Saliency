import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import *
from rasterize import batch_rasterize_relative
from base_model import Photo2Sketch_Base
from torchvision.utils import save_image
from Sketch_Networks import *
import os
from utils import Visualizer
# device = 'cpu'

class Photo2Sketch(Photo2Sketch_Base):
    def __init__(self, hp):

        Photo2Sketch_Base.__init__(self, hp)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate, betas=(0.5, 0.999))
        self.visualizer = Visualizer(hp.saved_models)

    def Image2Sketch_Train(self, rgb_image, sketch_vector, length_sketch, step, sketch_name):

        self.train()
        self.optimizer.zero_grad()

        curr_learning_rate = ((self.hp.learning_rate - self.hp.min_learning_rate) *
                              (self.hp.decay_rate) ** step + self.hp.min_learning_rate)
 
        """ Encoding the Input """
        backbone_feature, rgb_encoded_dist_z_vector = self.Image_Encoder(rgb_image)
         
        ##############################################################
        ##############################################################
        """ Cross Modal the Decoding """
        ##############################################################
        ##############################################################
        
        photo2sketch_output = self.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, sketch_vector, length_sketch + 1)
        
        end_token = torch.stack([torch.tensor([0, 0, 0, 0, 1])] * rgb_image.shape[0]).unsqueeze(0).to(device).float()
        batch = torch.cat([sketch_vector, end_token], 0)
        x_target = batch.permute(1, 0, 2)  # batch-> Seq_Len, Batch, Feature_dim
        
        sup_p2s_loss = sketch_reconstruction_loss(photo2sketch_output, x_target)  #TODO: Photo to Sketch Loss
        
        loss = sup_p2s_loss 
        
        set_learninRate(self.optimizer, curr_learning_rate)
        loss.backward()
        nn.utils.clip_grad_norm_(self.train_params, self.hp.grad_clip)
        self.optimizer.step()
        kl_cost_rgb=0
        print('Step:{} ** sup_p2s_loss:{} ** kl_cost_rgb:{} ** Total_loss:{}'.format(step, sup_p2s_loss,
                                                                               kl_cost_rgb, loss))

        if step%5 == 0:
        
            data = {}
            data['Reconstrcution_Loss'] = sup_p2s_loss
            data['KL_Loss'] = kl_cost_rgb
            data['Total Loss'] = loss
            self.visualizer.plot_scalars(data, step)

        
        if step%1 == 0:

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)

            with torch.no_grad():
                photo2sketch_gen, attention_plot  = \
                    self.Sketch_Decoder(backbone_feature, rgb_encoded_dist_z_vector, 
                                sketch_vector, length_sketch+1, isTrain=False)

            sketch_vector_gt = sketch_vector.permute(1, 0, 2)
            rgb_attention =  plot_attention(attention_plot, rgb_image)


            for num, len in enumerate(length_sketch):
                photo2sketch_gen[num, len:, 4 ] = 1.0
                photo2sketch_gen[num, len:, 2:4] = 0.0

            sketch_vector_gt_draw = batch_rasterize_relative(sketch_vector_gt)
            photo2sketch_gen_draw = batch_rasterize_relative(photo2sketch_gen)

            self.visualizer.vis_image({'sketch_redraw': photo2sketch_gen_draw,
                                        'sketch_vector_gt_draw':sketch_vector_gt_draw,
                                        'rgb_attention': rgb_attention}, step)

            # saved_folder_path = os.path.join(self.hp.saved_models, 'Redraw_Photo2Sketch')
            # os.makedirs(saved_folder_path, exist_ok=True)
            # save_image(photo2sketch_gen_draw, f'{saved_folder_path}/{step}_redraw.jpg',  
            #                              nrow=1, normalize=False)
            # save_image(sketch_vector_gt_draw, f'{saved_folder_path}/{step}_gt.jpg',
            #                              nrow=1, normalize=False)
 
        return sup_p2s_loss, kl_cost_rgb, loss