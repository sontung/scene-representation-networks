import configargparse
import os, time, datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import dataio
from torch.utils.data import DataLoader
from srns import *
import util
import time
import sys
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Multi-resolution training: Instead of passing only a single value, each of these command-line arguments take comma-
# separated lists. If no multi-resolution training is required, simply pass single values (see default values).
p.add_argument('--img_sidelengths', type=str, default='128', required=False,
               help='Progression of image sidelengths.'
                    'If comma-separated list, will train on each sidelength for respective max_steps.'
                    'Images are downsampled to the respective resolution.')
p.add_argument('--max_steps_per_img_sidelength', type=str, default="200000",
               help='Maximum number of optimization steps.'
                    'If comma-separated list, is understood as steps per image_sidelength.')
p.add_argument('--batch_size_per_img_sidelength', type=str, default="8",
               help='Training batch size.'
                    'If comma-separated list, will train each image sidelength with respective batch size.')

# Training options
p.add_argument('--data_root', required=True, help='Path to directory with training data.')
p.add_argument('--val_root', required=False, help='Path to directory with validation data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='path to directory where checkpoints & tensorboard events will be saved.')

p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')

p.add_argument('--l1_weight', type=float, default=200,
               help='Weight for l1 loss term (lambda_img in paper).')
p.add_argument('--kl_weight', type=float, default=1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--reg_weight', type=float, default=1e-3,
               help='Weight for depth regularization term (lambda_depth in paper).')

p.add_argument('--steps_til_ckpt', type=int, default=10000,
               help='Number of iterations until checkpoint is saved.')
p.add_argument('--steps_til_val', type=int, default=1000,
               help='Number of iterations until validation set is run.')
p.add_argument('--no_validation', action='store_true', default=False,
               help='If no validation set should be used.')

p.add_argument('--preload', action='store_true', default=False,
               help='Whether to preload data to RAM.')

p.add_argument('--checkpoint_path', default=None,
               help='Checkpoint to trained model.')
p.add_argument('--overwrite_embeddings', action='store_true', default=False,
               help='When loading from checkpoint: Whether to discard checkpoint embeddings and initialize at random.')
p.add_argument('--start_step', type=int, default=0,
               help='If continuing from checkpoint, which iteration to start counting at.')

p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')

p.add_argument('--max_num_instances_train', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances_train are used')
p.add_argument('--max_num_observations_train', type=int, default=50, required=False,
               help='If an instance has more observations, only the first max_num_observations_train are used')
p.add_argument('--max_num_instances_val', type=int, default=10, required=False,
               help='If \'val_root\' has more instances, only the first max_num_instances_val are used')
p.add_argument('--max_num_observations_val', type=int, default=10, required=False,
               help='Maximum numbers of observations per validation instance')

p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')

# Model options
p.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester.')
p.add_argument('--freeze_networks', action='store_true',
               help='Whether to freeze weights of all networks in SRN (not the embeddings!).')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')

opt = p.parse_args()


def train():
    # Parses indices of specific observations from comma-separated list.
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = util.parse_comma_separated_integers(opt.specific_observation_idcs)
    else:
        specific_observation_idcs = None

    img_sidelengths = util.parse_comma_separated_integers(opt.img_sidelengths)
    batch_size_per_sidelength = util.parse_comma_separated_integers(opt.batch_size_per_img_sidelength)
    max_steps_per_sidelength = util.parse_comma_separated_integers(opt.max_steps_per_img_sidelength)

    train_dataset = dataio.PBWDataset(train=True)

    assert (len(img_sidelengths) == len(batch_size_per_sidelength)), \
        "Different number of image sidelengths passed than batch sizes."
    assert (len(img_sidelengths) == len(max_steps_per_sidelength)), \
        "Different number of image sidelengths passed than max steps."

    if not opt.no_validation:
        assert (opt.val_root is not None), "No validation directory passed."

        val_dataset = dataio.PBWDataset(train=False)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=16,
                                    shuffle=False,
                                    drop_last=True,
                                    collate_fn=val_dataset.collate_fn)

    model = SRNsModel3(latent_dim=opt.embedding_size,
                       has_params=opt.has_params,
                       fit_single_srn=True,
                       tracing_steps=opt.tracing_steps,
                       freeze_networks=opt.freeze_networks)
    model.train()
    model.cuda()
    if opt.checkpoint_path is not None:
        print("Loading model from %s" % opt.checkpoint_path)
        util.custom_load(model, path=opt.checkpoint_path,
                         discriminator=None,
                         optimizer=None,
                         overwrite_embeddings=opt.overwrite_embeddings)

    ckpt_dir = os.path.join(opt.logging_root, 'checkpoints')
    events_dir = os.path.join(opt.logging_root, 'events')

    util.cond_mkdir(opt.logging_root)
    util.cond_mkdir(ckpt_dir)
    util.cond_mkdir(events_dir)

    # Save command-line parameters log directory.
    with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    # Save text summary of model into log directory.
    with open(os.path.join(opt.logging_root, "model.txt"), "w") as out_file:
        out_file.write(str(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    writer = SummaryWriter(events_dir)
    iter = opt.start_step
    epoch = iter // len(train_dataset)
    step = 0

    print('Beginning training...')
    # This loop implements training with an increasing image sidelength.
    cum_max_steps = 0  # Tracks max_steps cumulatively over all image sidelengths.
    for img_sidelength, max_steps, batch_size in zip(img_sidelengths, max_steps_per_sidelength,
                                                     batch_size_per_sidelength):
        print("\n" + "#" * 10)
        print("Training with sidelength %d for %d steps with batch size %d" % (img_sidelength, max_steps, batch_size))
        print("#" * 10 + "\n")

        # Need to instantiate DataLoader every time to set new batch size.
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      collate_fn=train_dataset.collate_fn,)

        cum_max_steps += max_steps

        # Loops over epochs.
        while True:
            for batch in train_dataloader:
                rgb, ext_mat, info, rgb_mat = batch
                ground_truth = {"rgb": rgb}
                model_input = (ext_mat, rgb_mat, info) # color, pix coord, location, box
                model_outputs = model(model_input)
                optimizer.zero_grad()

                total_loss = model.get_image_loss(model_outputs, ground_truth)
                total_loss.backward()

                optimizer.step()
                if iter % 100 == 0:
                    print("Iter %07d   Epoch %03d   L_img %0.4f" %
                          (iter, epoch, total_loss))


                if iter % opt.steps_til_val == 0 and not opt.no_validation:
                    print("Running validation set...")
                    acc = test(model, val_dataloader, str(iter))
                    print("Accuracy:", acc)

                iter += 1
                step += 1

                if iter == cum_max_steps:
                    break

            if iter == cum_max_steps:
                break
            epoch += 1

    util.custom_save(model,
                     os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                     discriminator=None,
                     optimizer=optimizer)

def show2(im_, name, nrow):
    import logging

    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig_ = plt.figure(figsize=(15, 15))
    for du3 in range(1, len(im_)+1):
        plt.subplot(1, len(im_), du3)
        plt.axis("off")
        plt.imshow(np.transpose(make_grid(im_[du3-1], padding=5, normalize=False, pad_value=50, nrow=nrow),
                                (1, 2, 0)))

    plt.axis("off")
    # plt.title("black: no action, red: 1-3, yellow: 3-1, green: 1-2, blue: 2-3, pink: 3-2, brown: 2-1")
    plt.savefig(name, transparent=True, bbox_inches='tight')
    plt.close(fig_)
    logger.setLevel(old_level)

def test(model_, loader_, name_):
    model_.eval()
    acc = 0.0
    nb_samples = 0
    save_im = False
    with torch.no_grad():
        for batch in loader_:
            rgb, ext_mat, info, rgb_mat = batch
            model_input = (ext_mat, rgb_mat, info)  # color, pix coord, location, box

            all_arrangements = []
            all_uv = []
            for scene in info:
                true_obj_arrangement = [(obj[0], obj[1]) for obj in scene]
                uv = [obj[1][:2] for obj in scene]
                all_arrangements.append(true_obj_arrangement)
                all_uv.append(uv)

            world_coords = model_.return_world_coords(torch.tensor(all_uv), model_input)

            for i in range(len(all_arrangements)):
                scene = all_arrangements[i]
                pred_wc = world_coords[i]
                true = []
                pred = []
                all_arr = []
                for j, obj in enumerate(scene):
                    true.append(obj[1][-1])
                    pred.append(pred_wc[j][2].item())
                    all_arr.append((obj[0], obj[1][-1], pred_wc[j][2].item()))
                # print("true:", sorted(all_arr, key=lambda du: du[1]), "\npred:",
                #       sorted(all_arr, key=lambda du: du[2]))
                true_rel = sorted(all_arr, key=lambda du: du[1])
                pred_rel = sorted(all_arr, key=lambda du: du[2])
                if true_rel == pred_rel:
                    acc +=1
                nb_samples += 1

            if not save_im:
                save_im = True
                rgb_pred = model_(model_input)
                show2([
                    rgb.reshape(rgb.size(0), 128, 128, 3).permute(0, 3, 1, 2).cpu(),
                    rgb_pred.reshape(rgb.size(0), 128, 128, 3).permute(0, 3, 1, 2).cpu(),
                ], "figures/test%s.png" % name_, 4)

    model_.train()
    return acc / nb_samples

def main():
    train()


if __name__ == '__main__':
    main()
