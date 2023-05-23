import numpy as np
from bresenham import bresenham
import scipy.ndimage
from PIL import Image
import torch


def get_stroke_num(vector_image):
    return len(np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1])


def mydrawPNG_fromlist(vector_image, stroke_idx, Side=256):
    vector_image = np.split(vector_image[:, :2], np.where(vector_image[:, 2])[0] + 1, axis=0)[:-1]
    vector_image = [vector_image[x] for x in stroke_idx]

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
            initX, initY = int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return Image.fromarray(raster_image).convert('RGB')


def mydrawPNG(vector_image, Side=256):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0
    return raster_image


def preprocess(sketch_points,  side_norm=800, side=256.0):
    sketch_points = sketch_points.astype(np.float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([side_norm, side_norm])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points


def rasterize_Sketch(sketch_points, side_norm=800):
    sketch_points = preprocess(sketch_points, side_norm)
    raster_images = mydrawPNG(sketch_points)
    return raster_images, sketch_points

def mydrawPNG_from_list(vector_image, Side = 256):

    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)

    for stroke in vector_image:
        initX, initY = int(stroke[0, 0]), int(stroke[0, 1])

        for i_pos in range(1, len(stroke)):
            cordList = list(bresenham(initX, initY, int(stroke[i_pos, 0]), int(stroke[i_pos, 1])))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] <= Side and cord[1] <= Side):
                    raster_image[cord[1], cord[0]] = 255.0
                else:
                    print('error')
            initX, initY =  int(stroke[i_pos, 0]), int(stroke[i_pos, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image) * 255.0

    return Image.fromarray(raster_image).convert('RGB')

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

def batch_rasterize_relative(sketch):

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

    batch_redraw = []
    if sketch.shape[-1] == 5:
        for data in sketch:
            # image = rasterize_relative(to_stroke_list(to_normal_strokes(data.cpu().numpy())), canvas)
            image = mydrawPNG_from_list(to_stroke_list(to_normal_strokes(data.cpu().numpy())))
            batch_redraw.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    elif sketch.shape[-1] == 3:
        for data in sketch:
            # image = rasterize_relative(to_stroke_list(data.cpu().numpy()), canvas)
            image = mydrawPNG_from_list(to_stroke_list(data.cpu().numpy()))
            batch_redraw.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))

    return torch.stack(batch_redraw).float()
