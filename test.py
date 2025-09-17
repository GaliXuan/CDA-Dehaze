import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torch.nn.functional as F
import util.util as util
import numpy as np
from PIL import Image

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0
    return 10 * torch.log10(max_pixel_value ** 2 / mse)

def read_images_folder1(folder_path):
    image_tensors_folder1 = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
            image_tensors_folder1.append(image_tensor)
    return image_tensors_folder1

def read_images_folder2(folder_path):
    image_tensors_folder2 = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
            image_tensors_folder2.append(image_tensor)
    return image_tensors_folder2

def calculate_average_psnr(folder_path1, folder_path2):
    image_tensors_folder1 = read_images_folder1(folder_path1)
    image_tensors_folder2 = read_images_folder2(folder_path2)
    assert len(image_tensors_folder1) == len(image_tensors_folder2)
    total_psnr = 0
    num_pairs = len(image_tensors_folder1)
    for i in range(num_pairs):
        psnr_value = calculate_psnr(image_tensors_folder1[i], image_tensors_folder2[i])
        total_psnr += psnr_value
    return total_psnr / num_pairs

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    # opt.epoch=34

    model = create_model(opt)  # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    total_psnr = 0

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    print('Test End!!!!!!!!!!!!!!!')

