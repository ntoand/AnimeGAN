import matplotlib.pyplot as plt
import torch
import numpy as np
from model_utils import Discriminator, Generator
import argparse
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='output', help='Specify the output dir to save the images')
    parser.add_argument('--output-name', default='chararter', help='Specify output name (no extention)')
    parser.add_argument('--num-images', type=int, default=1, help='number of characters to be created')
    options = parser.parse_args()
    return options

def create_directories(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_model():
    model = Generator()
    model.load_state_dict(torch.load('Generator.pth', map_location=torch.device('cpu')))
    return model

def generate(model, num_images):
    images = model(torch.randn(num_images, 100, 1, 1))
    return images.detach().numpy()

def save_anime(opts, images):
    for i in range(len(images)):
        anime = images[i].reshape(3, 64, 64)
        anime = np.moveaxis(anime, 0, 2)
        anime = anime * np.array((0.5, 0.5, 0.5)) + np.array((0.5,0.5,0.5))
        anime = np.clip(anime, 0, 1)
        plt.imsave(opts.output_dir + f'/{opts.output_name}{i+1}.png', anime)
        print('Saved image to {}'.format(opts.output_dir + f'/{opts.output_name}{i+1}.png'))

if __name__ == '__main__':
    arguments = get_arguments()
    create_directories(arguments.output_dir)
    model = load_model()
    print('Generating Characters...')
    images = generate(model, int(arguments.num_images))
    print('Saving Characters...')
    save_anime(arguments, images)
    print('Done')
