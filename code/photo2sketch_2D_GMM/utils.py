import torchvision
import os
from rasterize import mydrawPNG
from PIL import Image
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import shutil
import cv2
import imageio
from matplotlib import pyplot as plt
from rasterize import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image

def to_delXY(sketch):
    new_skech = sketch.copy()
    new_skech[:-1,:2]  = new_skech[1:,:2] - new_skech[:-1,:2]
    new_skech[:-1, 2] = new_skech[1:, 2]
    return new_skech[:-1,:]



def to_Five_Point(sketch_points, max_seq_length):
    len_seq = len(sketch_points[:, 0])
    new_seq = np.zeros((max_seq_length, 5))
    new_seq[0:len_seq, :2] = sketch_points[:, :2]
    new_seq[0:len_seq, 3] = sketch_points[:, 2]
    new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
    new_seq[(len_seq - 1):, 4] = 1
    new_seq[(len_seq - 1), 2:4] = 0
    new_seq = np.concatenate((np.zeros((1, 5)), new_seq), axis=0)
    return new_seq, len_seq

def to_stroke_list(sketch):
    ## sketch: an `.npz` style sketch from QuickDraw
    sketch = np.vstack((np.array([0, 0, 0]), sketch))
    sketch[:, :2] = np.cumsum(sketch[:, :2], axis=0)

    # range normalization
    xmin, xmax = sketch[:, 0].min(), sketch[:, 0].max()
    ymin, ymax = sketch[:, 1].min(), sketch[:, 1].max()

    sketch[:, 0] = ((sketch[:, 0] - xmin) / float(xmax - xmin)) * (255. - 60.) + 30.
    sketch[:, 1] = ((sketch[:, 1] - ymin) / float(ymax - ymin)) * (255. - 60.) + 30.
    sketch = sketch.astype(np.int64)

    stroke_list = np.split(sketch[:, :2], np.where(sketch[:, 2])[0] + 1, axis=0)

    if stroke_list[-1].size == 0:
        stroke_list = stroke_list[:-1]

    if len(stroke_list) == 0:
        stroke_list = [sketch[:, :2]]
        # print('error')
    return stroke_list

def eval_redraw(target, output, seq_len, step, saved_folder, type, num_print=8, side=1):

    batch_redraw = []

    for sample_targ, sample_gen, seq in zip(target[:num_print], output[:num_print], seq_len[:num_print]):

        sample_gen = sample_gen.cpu().numpy()[:seq]
        sample_targ = sample_targ.cpu().numpy()
        sample_targ = to_normal_strokes(sample_targ)
        sample_gen = to_normal_strokes(sample_gen)

        sample_gen[:, :2] = np.round(sample_gen[:, :2] * side)
        image_gen = mydrawPNG(sample_gen)
        image_gen = Image.fromarray(image_gen).convert('RGB')

        sample_targ[:, :2] = np.round(sample_targ[:, :2] * side)
        image_targ = mydrawPNG(sample_targ)
        image_targ = Image.fromarray(image_targ).convert('RGB')

        batch_redraw.append(torch.from_numpy(np.array(image_targ)).permute(2, 0, 1))
        batch_redraw.append(torch.from_numpy(np.array(image_gen)).permute(2, 0, 1))

    batch_redraw = torch.stack(batch_redraw).float()
    torchvision.utils.save_image(batch_redraw, os.path.join(
        saved_folder, type + '_' + str(step) + '.jpg'), normalize=True, nrow=2)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    result[-1, 2] = 1.
    return result

def save_sketch_gen(sketch_vector, sketch_name, saved_models, mode):
    stroke_list = to_stroke_list(to_normal_strokes(sketch_vector.cpu().numpy()))

    folder_name = os.path.join('CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]), sketch_name.split('/')[-1]+mode)
    folder_name = os.path.join(saved_models, folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    fig = plt.figure(frameon=False, figsize=(2.56, 2.56))
    xlim = [0, 255]
    ylim = [0, 255]
    x_count = 0
    count = 0
    for stroke in stroke_list:
        stroke_buffer = np.array(stroke[0])
        for x_num in range(len(stroke)):
            x_count = x_count + 1
            stroke_buffer = np.vstack((stroke_buffer, stroke[x_num, :2]))
            if x_count % 5 == 0:

                plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)
                plt.gca().invert_yaxis();
                plt.axis('off')

                plt.savefig(folder_name + '/sketch_' + str(count) + 'points_.jpg', bbox_inches='tight',
                            pad_inches=0, dpi=1200)
                count = count + 1
                plt.gca().invert_yaxis();


        plt.plot(stroke_buffer[:, 0], stroke_buffer[:, 1], '.', linestyle='solid', linewidth=1.0, markersize=5)



def plot_attention(attention_plot, photo_image):
    alpha = 0.5
    photo_attention = []
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to('cuda')
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to('cuda')
    photo_image = photo_image.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    for i_num, x_photo in enumerate(photo_image):

        x_photo = x_photo.permute(1, 2, 0).cpu().numpy()
        x_photo = cv2.resize(np.float32(np.uint8(255. * x_photo)), (256, 256))
 
        attention = [x[i_num] for x in attention_plot]
 
        # attention = torch.stack(attention).sum(dim=0).squeeze()
        # attention = F.softmax(attention.view(-1), dim=-1).reshape(8,8)
        attention = torch.stack(attention).mean(0).squeeze()
        attention = attention/attention.max()
        attention = attention.cpu().numpy()
 
        # attention[attention < 0.01] = 0
        # attention = attention / attention.sum()
        # attention = np.clip(attention / attention.max() * 255., 0, 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
        heatmap = cv2.resize(np.float32(heatmap), (256, 256))

        heat_map_overlay = cv2.addWeighted(heatmap, alpha, x_photo, 1 - alpha, 0)
        heat_map_tensor = torch.from_numpy(heat_map_overlay).permute(2, 0, 1)/255.
        photo_attention.append(heat_map_tensor)

    # save_image(torch.stack(photo_attention), 'photo_attention.jpg', normalize=True)

    return torch.stack(photo_attention) 

def showAttention(attention_plot, sketch_img, sketch_vector_gt_draw, photo2sketch_gen_draw, sketch_name, saved_models):
    # Set up figure with colorbar
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to('cpu')
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to('cpu')


    folder_name = os.path.join('./CVPR_SSL/' + '_'.join(sketch_name.split('/')[-1].split('_')[:-1]))
    folder_name = os.path.join(saved_models, folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    # sketch_vector_gt_draw = sketch_vector_gt_draw.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    sketch_vector_gt_draw = sketch_vector_gt_draw.squeeze(0).permute(1, 2, 0).numpy()
    sketch_vector_gt_draw = cv2.resize(np.float32(np.uint8(255. * (1. - sketch_vector_gt_draw))), (256, 256))

    # photo2sketch_gen_draw = photo2sketch_gen_draw.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    photo2sketch_gen_draw = photo2sketch_gen_draw.squeeze(0).permute(1, 2, 0).numpy()
    photo2sketch_gen_draw = cv2.resize(np.float32(np.uint8(255. * (1. -photo2sketch_gen_draw))), (256, 256))

    imageio.imwrite(folder_name + '/sketch_' + 'GT.jpg', sketch_vector_gt_draw)
    imageio.imwrite(folder_name + '/sketch_' + 'Gen.jpg', photo2sketch_gen_draw)

    attention_dictionary = {}
    for num, val in enumerate(sketch_img):
        attention_dictionary[num] = []
        val = val.cpu()
        x = val.unsqueeze(0)
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        x = x.squeeze(0)
        attention_dictionary[num].append(x)


    alpha = 0.5
    for atten_num, x_data in enumerate(attention_plot):
        for num, per_image_x in enumerate(x_data):

            attention = per_image_x.squeeze(0).cpu().numpy()

            # attention[attention < 0.01] = 0
            # attention = attention / attention.sum()
            # attention = np.clip(attention / attention.max() * 255., 0, 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
            heatmap = cv2.resize(np.float32(heatmap), (256, 256))

            # heatmap = heatmap**2


            # image = 255. - attention_dictionary[num][0].permute(1, 2, 0).numpy()

            # mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).to('cpu')
            # std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).to('cpu')
            # x = attention_dictionary[num][0].unsqueeze(0)
            # x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            # x = x.squeeze(0)
            image = attention_dictionary[num][0].permute(1, 2, 0).numpy()
            image = cv2.resize(np.float32(np.uint8(255. * image)), (256, 256))

            # image preprocess

            imageio.imwrite(folder_name  + '/RGB.jpg', image)

            heat_map_overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
            imageio.imwrite(folder_name + '/' + sketch_name.split('/')[-1] +  '_' + str(atten_num) + '.jpg', heat_map_overlay)


            heat_map_tensor = torch.from_numpy(heat_map_overlay).permute(2, 0, 1)
            # heat_map_tensor = attention_dictionary[num][0] + torch.from_numpy(heatmap).permute(2, 0, 1)
            # heat_map_tensor = heat_map_tensor / heat_map_tensor.max()
            attention_dictionary[num].append(heat_map_tensor)

    # plot_attention = []
    # for num, val in enumerate(sketch_img):
    #     image = torch.stack(attention_dictionary[num][1:], dim=0).permute(1, 2, 0, 3).reshape(3, 256, -1)
    #     image.add_(-image.min()).div_(image.max() - image.min()+ 1e-5)
    #     plot_attention.append(image)

    # return torch.stack(plot_attention)
    return None

class Visualizer:
    def __init__(self, folder = 'Photo2Sketch2D_logs'):

        folder = f'{folder}/TensorBoard_logs/'

        if os.path.exists(folder):
            shutil.rmtree(folder)
 
        self.writer = SummaryWriter(folder, flush_secs=10)
        self.mean = torch.tensor([-1.0, -1.0, -1.0]).to(device)
        self.std = torch.tensor([1 / 0.5, 1 / 0.5, 1 / 0.5]).to(device)

    def vis_image(self, visularize, step, normalize=False):
        for keys, value in visularize.items():
            #print(keys,value.size())
            if normalize:
                value.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
            visularize[keys] = torchvision.utils.make_grid(value, normalize=True)
            self.writer.add_image('{}'.format(keys), visularize[keys], step)


    def plot_scalars(self, losses, step):
        self.writer.add_scalars('Training Losses', losses, step)