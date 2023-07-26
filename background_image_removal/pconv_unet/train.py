import os
import glob

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import PConvUnet
from utils import MaskGenerator
from data_input_pipeline import get_dataset


def generate_mask(batch_size, generator_obj):
    return np.stack([
        generator_obj.sample()
        for _ in range(batch_size)], axis=0
    ) * 1.0


def get_masked(batch, generator_obj):
    masks = generate_mask(batch.shape[0], generator_obj).astype("float32")
    return masks, batch


def plot_example(inp, tar, gen, comp, train_dir, epoch, step):
    plot_dir = f"{train_dir}/plots/"
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 4, sharex=True, figsize=(16.5, 16.5))

    ax[0].imshow(inp)
    ax[0].set_title('Input')

    ax[1].imshow(gen)
    ax[1].set_title('Generated')

    ax[2].imshow(tar)
    ax[2].set_title('Target')

    ax[3].imshow(comp)
    ax[3].set_title('Computed')

    plt.savefig(f"{plot_dir}/e{epoch}_s{step}.png")
    plt.close()


def forward(model, target, mask):
    gen_output = model.pconv_unet([target, mask])
    comp = target * mask + (1 - mask) * gen_output

    vgg_real = model.vgg(target)
    vgg_gen = model.vgg(gen_output)
    vgg_comp = model.vgg(comp)

    gen_losses, losses_dict = model.gen_loss(
        vgg_real, vgg_gen, vgg_comp, target, gen_output, mask)

    return gen_output, comp, gen_losses, losses_dict


def train(model, train_iter, valid_iter, train_dir, start_epoch=0, epochs=10, train_steps=100000, valid_steps=1000, plot_interval=100, image_size=256):
    os.makedirs(train_dir, exist_ok=True)
    mask_gen = MaskGenerator(image_size, image_size)
    for e in range(start_epoch, epochs):
        train_losses = []
        valid_losses = []
        print(f"Epoch {e} - training")
        pbar = tqdm(range(train_steps))
        for i in pbar:
            batch = next(train_iter)
            mask, target = get_masked(batch, mask_gen)
            with tf.GradientTape() as gen_tape:
                gen_output, comp, gen_losses, losses_dict = forward(
                    model, target, mask)
                train_losses.append(losses_dict["total"])
                pbar.set_postfix({"loss": train_losses[-1]})
                for k in losses_dict.keys():
                    if k in model.losses.keys():
                        model.losses[k].append(losses_dict[k].numpy() * (1 - model.ema) +
                                               model.losses[k][-1] * model.ema)
                    else:
                        model.losses[k] = [losses_dict[k].numpy()]

                gen_grads = gen_tape.gradient(
                    gen_losses, model.pconv_unet.trainable_variables)
                model.optimizer.apply_gradients(
                    zip(gen_grads, model.pconv_unet.trainable_variables))

        print(f"Epoch {e} - validation")
        pbar = tqdm(range(valid_steps))
        for i in pbar:
            batch = next(valid_iter)
            mask, target = get_masked(batch, mask_gen)
            gen_output, comp, gen_losses, losses_dict = forward(
                model, target, mask)
            valid_losses.append(losses_dict["total"])
            pbar.set_postfix({"loss": valid_losses[-1]})
            if plot_interval and i % plot_interval == 0:
                plot_example(target[0] * mask[0], target[0],
                             gen_output[0], comp[0], train_dir, e, i)
        model.save_model(
            os.path.join(
                train_dir,
                f"epoch_{e}_train_loss_tl{np.mean(train_losses)}_vl{np.mean(valid_losses)}"))


if __name__ == "__main__":
    image_size = 256
    batch_size = 16
    assets_dir = f"{os.path.dirname(__file__)}/assets"
    model_ckpt = f"{assets_dir}/pytorch_to_keras_vgg16.h5"
    train_filepaths = glob.glob(
        f"{assets_dir}/kaggle/**/train*-of-01024", recursive=True)
    valid_filepaths = glob.glob(
        f"{assets_dir}/kaggle/**/validation*-of-0128", recursive=True)
    if not valid_filepaths:
        train_filepaths = train_filepaths[:-1]
        valid_filepaths = train_filepaths[-1:]
    train_ds = get_dataset(train_filepaths, image_size,
                           batch_size, training=True)
    valid_ds = get_dataset(valid_filepaths, image_size,
                           batch_size, training=False)
    model = PConvUnet(model_ckpt, image_size=image_size, lr=1e-2)

    train_dir = f"{assets_dir}/trained_models"
    start_epoch = 0
    previous_ckpt = tf.train.latest_checkpoint(
        train_dir, f"{train_dir}/checkpoint")
    if previous_ckpt:
        model.load_model(previous_ckpt)
        start_epoch = int(previous_ckpt.split("/")[-1].split("_")[1]) + 1
        print(f"Model weights loaded from {previous_ckpt}")
        print(f"Start training at epoch {start_epoch}")

    # Full-on training
    train(
        model,
        train_iter=iter(train_ds),
        valid_iter=iter(valid_ds),
        train_dir=train_dir,
        start_epoch=start_epoch,
        epochs=100,
        train_steps=10000,
        valid_steps=100,
        plot_interval=20,
        image_size=256
    )
