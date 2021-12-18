import torch
from models import LatentEBM, ToyEBM, BetaVAE_H, LatentEBM128
from tensorflow.python.platform import flags
import torch.nn.functional as F
import os
from dataset import IntPhysDataset, ToyDataset, TFImagenetLoader, CubesColor, CubesColorPair, TFTaskAdaptation, DSprites, Blender, Cub, Nvidia, Clevr, Exercise, CelebaHQ, Kitti, Airplane, Faces, ClevrLighting
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from easydict import EasyDict
import os.path as osp
from torch.nn.utils import clip_grad_norm
import numpy as np
from imageio import imwrite
import cv2
import argparse
import pdb
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import make_grid
from dataset import MultiDspritesLoader, TetrominoesLoader
from imageio import get_writer


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--optimize_test', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')
parser.add_argument('--single', action='store_true', help='test overfitting of the dataset')


parser.add_argument('--dataset', default='blender', type=str, help='Dataset to use (intphys or others or imagenet or cubes)')
parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='default', type=str, help='name of experiments')

# training
parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
parser.add_argument('--batch_size', default=64, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')
parser.add_argument('--ensembles', default=1, type=int, help='use an ensemble of models')
parser.add_argument('--vae-beta', type=float, default=0.)

# EBM specific settings

# Model specific settings
parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
parser.add_argument('--components', default=2, type=int, help='number of components to explain an image with')
parser.add_argument('--component_weight', action='store_true', help='optimize for weights of the components also')
parser.add_argument('--tie_weight', action='store_true', help='tie the weights between seperate models')
parser.add_argument('--optimize_mask', action='store_true', help='also optimize a segmentation mask over image')
parser.add_argument('--recurrent_model', action='store_true', help='use a recurrent model to infer latents')
parser.add_argument('--pos_embed', action='store_true', help='add a positional embedding to model')
parser.add_argument('--spatial_feat', action='store_true', help='use spatial latents for object segmentation')


parser.add_argument('--num_steps', default=10, type=int, help='Steps of gradient descent for training')
parser.add_argument('--num_visuals', default=16, type=int, help='Number of visuals')
parser.add_argument('--num_additional', default=0, type=int, help='Number of additional components to add')

parser.add_argument('--step_lr', default=500.0, type=float, help='step size of latents')

parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--sample', action='store_true', help='generate negative samples through Langevin')
parser.add_argument('--decoder', action='store_true', help='decoder for model')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')



def average_gradients(models):
    size = float(dist.get_world_size())

    for model in models:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size


def gen_image(latents, FLAGS, models, im_neg, im, num_steps, sample=False, create_graph=True, idx=None, weights=None):
    im_noise = torch.randn_like(im_neg).detach()
    im_negs_samples = []

    im_negs = []

    latents = torch.stack(latents, dim=0)

    if FLAGS.decoder:
        masks = []
        colors = []
        for i in range(len(latents)):
            if idx is not None and idx != i:
                pass
            else:
                color, mask = models[i % FLAGS.components].forward(None, latents[i])
                masks.append(mask)
                colors.append(color)
        masks = F.softmax(torch.stack(masks, dim=1), dim=1)
        colors = torch.stack(colors, dim=1)
        im_neg = torch.sum(masks * colors, dim=1)
        im_negs = [im_neg]
        im_grad = torch.zeros_like(im_neg)
    else:
        im_neg.requires_grad_(requires_grad=True)
        s = im.size()
        masks = torch.zeros(s[0], FLAGS.components, s[-2], s[-1]).to(im_neg.device)
        masks.requires_grad_(requires_grad=True)

        for i in range(num_steps):
            im_noise.normal_()

            energy = 0
            for j in range(len(latents)):
                if idx is not None and idx != j:
                    pass
                else:
                    ix = j % FLAGS.components
                    energy = models[j % FLAGS.components].forward(im_neg, latents[j]) + energy

            im_grad, = torch.autograd.grad([energy.sum()], [im_neg], create_graph=create_graph)

            im_neg = im_neg - FLAGS.step_lr * im_grad

            latents = latents

            im_neg = torch.clamp(im_neg, 0, 1)
            im_negs.append(im_neg)
            im_neg = im_neg.detach()
            im_neg.requires_grad_()

    return im_neg, im_negs, im_grad, masks


def ema_model(models, models_ema, mu=0.999):
    for (model, model_ema) in zip(models, models_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(models):
    size = float(dist.get_world_size())

    for model in models:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(FLAGS, device, dataset):
    if FLAGS.tie_weight:
        if FLAGS.dataset == "toy":
            model = ToyEBM(FLAGS, dataset).to(device)
        else:
            if FLAGS.vae_beta:
                model = BetaVAE_H(z_dim=FLAGS.latent_dim, nc=3).to(device)
                FLAGS.ensembles = 1
                FLAGS.components = 1
            else:
                if FLAGS.dataset == "celebahq_128":
                    model = LatentEBM128(FLAGS, dataset).to(device)
                else:
                    model = LatentEBM(FLAGS, dataset).to(device)

        models = [model for i in range(FLAGS.ensembles)]
        optimizers = [Adam(model.parameters(), lr=FLAGS.lr)]
    else:
        models = [LatentEBM(FLAGS, dataset).to(device) for i in range(FLAGS.ensembles)]

        optimizers = [Adam(model.parameters(), lr=FLAGS.lr) for model in models]

    return models, optimizers


def test(train_dataloader, models, FLAGS, step=0):
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None

    [model.eval() for model in models]
    for im, idx in train_dataloader:

        im = im.to(dev)
        idx = idx.to(dev)
        im = im[:FLAGS.num_visuals]
        idx = idx[:FLAGS.num_visuals]
        batch_size = im.size(0)
        latent = models[0].embed_latent(im)

        latents = torch.chunk(latent, FLAGS.components, dim=1)

        im_init = torch.rand_like(im)
        assert len(latents) == FLAGS.components
        im_neg, _, im_grad, mask = gen_image(latents, FLAGS, models, im_init, im, FLAGS.num_steps, sample=FLAGS.sample, 
                                       create_graph=False)
        im_neg = im_neg.detach()
        im_components = []

        if FLAGS.components > 1:
            for i, latent in enumerate(latents):
                im_init = torch.rand_like(im)
                latents_select = latents[i:i+1]
                im_component, _, _, _ = gen_image(latents_select, FLAGS, models, im_init, im, FLAGS.num_steps, sample=FLAGS.sample,
                                           create_graph=False)
                im_components.append(im_component)

            im_init = torch.rand_like(im)
            latents_perm = [torch.cat([latent[i:], latent[:i]], dim=0) for i, latent in enumerate(latents)]
            im_neg_perm, _, im_grad_perm, _ = gen_image(latents_perm, FLAGS, models, im_init, im, FLAGS.num_steps, sample=FLAGS.sample,
                                                     create_graph=False)
            im_neg_perm = im_neg_perm.detach()
            im_init = torch.rand_like(im)
            add_latents = list(latents)
            for i in range(FLAGS.num_additional):
                add_latents.append(torch.roll(latents[i], i + 1, 0))
            im_neg_additional, _, _, _ = gen_image(tuple(add_latents), FLAGS, models, im_init, im, FLAGS.num_steps, sample=FLAGS.sample,
                                                     create_graph=False)

        im.requires_grad = True
        im_grads = []

        for i, latent in enumerate(latents):
            if FLAGS.decoder:
                im_grad = torch.zeros_like(im)
            else:
                energy_pos = models[i].forward(im, latents[i])
                im_grad = torch.autograd.grad([energy_pos.sum()], [im])[0]
            im_grads.append(im_grad)

        im_grad = torch.stack(im_grads, dim=1)

        s = im.size()
        im_size = s[-1]

        im_grad = im_grad.view(batch_size, FLAGS.components, 3, im_size, im_size) # [4, 3, 3, 128, 128]
        im_grad_dense = im_grad.view(batch_size, FLAGS.components, 1, 3 * im_size * im_size, 1) # [4, 3, 1, 49152, 1]
        im_grad_min = im_grad_dense.min(dim=3, keepdim=True)[0]
        im_grad_max = im_grad_dense.max(dim=3, keepdim=True)[0] # [4, 3, 1, 1, 1]


        im_grad = (im_grad - im_grad_min) / (im_grad_max - im_grad_min + 1e-5) # [4, 3, 3, 128, 128]
        im_grad[:, :, :, :1, :] = 1
        im_grad[:, :, :, -1:, :] = 1
        im_grad[:, :, :, :, :1] = 1
        im_grad[:, :, :, :, -1:] = 1
        im_output = im_grad.permute(0, 3, 1, 4, 2).reshape(batch_size * im_size, FLAGS.components * im_size, 3)
        im_output = im_output.cpu().detach().numpy() * 100

        im_output = (im_output - im_output.min()) / (im_output.max() - im_output.min())

        im = im.cpu().detach().numpy().transpose((0, 2, 3, 1)).reshape(batch_size*im_size, im_size, 3)

        im_output = np.concatenate([im_output, im], axis=1)
        im_output = im_output*255
        imwrite("result/%s/s%08d_grad.png" % (FLAGS.exp,step), im_output)

        im_neg = im_neg_tensor = im_neg.detach().cpu()
        im_components = [im_components[i].detach().cpu() for i in range(len(im_components))]
        im_neg = torch.cat([im_neg] + im_components)
        im_neg = np.clip(im_neg, 0.0, 1.0)
        im_neg = make_grid(im_neg, nrow=int(im_neg.shape[0] / (FLAGS.components + 1))).permute(1, 2, 0)
        im_neg = im_neg.numpy()*255
        imwrite("result/%s/s%08d_gen.png" % (FLAGS.exp,step), im_neg)

        if FLAGS.components > 1:
            im_neg_perm = im_neg_perm.detach().cpu()
            im_components_perm = []
            for i,im_component in enumerate(im_components):
                im_components_perm.append(torch.cat([im_component[i:], im_component[:i]]))
            im_neg_perm = torch.cat([im_neg_perm] + im_components_perm)
            im_neg_perm = np.clip(im_neg_perm, 0.0, 1.0)
            im_neg_perm = make_grid(im_neg_perm, nrow=int(im_neg_perm.shape[0] / (FLAGS.components + 1))).permute(1, 2, 0)
            im_neg_perm = im_neg_perm.numpy()*255
            imwrite("result/%s/s%08d_gen_perm.png" % (FLAGS.exp,step), im_neg_perm)

            im_neg_additional = im_neg_additional.detach().cpu()
            for i in range(FLAGS.num_additional):
                im_components.append(torch.roll(im_components[i], i + 1, 0))
            im_neg_additional = torch.cat([im_neg_additional] + im_components)
            im_neg_additional = np.clip(im_neg_additional, 0.0, 1.0)
            im_neg_additional = make_grid(im_neg_additional, 
                                nrow=int(im_neg_additional.shape[0] / (FLAGS.components + FLAGS.num_additional + 1))).permute(1, 2, 0)
            im_neg_additional = im_neg_additional.numpy()*255
            imwrite("result/%s/s%08d_gen_add.png" % (FLAGS.exp,step), im_neg_additional)

            print('test at step %d done!' % step)
        break

    [model.train() for model in models]


def train(train_dataloader, test_dataloader, logger, models, optimizers, FLAGS, logdir, rank_idx):
    it = FLAGS.resume_iter
    [optimizer.zero_grad() for optimizer in optimizers]

    dev = torch.device("cuda")

    # Use LPIPS loss for CelebA-HQ 128x128
    if FLAGS.dataset == "celebahq_128":
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    for epoch in range(FLAGS.num_epoch):
        for im, idx in train_dataloader:

            im = im.to(dev)
            idx = idx.to(dev)
            im_orig = im

            random_idx = random.randint(0, FLAGS.ensembles - 1)
            random_idx = 0

            latent = models[0].embed_latent(im)

            latents = torch.chunk(latent, FLAGS.components, dim=1)

            im_neg = torch.rand_like(im)
            im_neg_init = im_neg

            im_neg, im_negs, im_grad, _ = gen_image(latents, FLAGS, models, im_neg, im, FLAGS.num_steps, FLAGS.sample)

            im_negs = torch.stack(im_negs, dim=1)

            energy_pos = 0
            energy_neg = 0

            energy_poss = []
            energy_negs = []
            for i in range(FLAGS.components):
                energy_poss.append(models[i].forward(im, latents[i]))
                energy_negs.append(models[i].forward(im_neg.detach(), latents[i]))

            energy_pos = torch.stack(energy_poss, dim=1)
            energy_neg = torch.stack(energy_negs, dim=1)
            ml_loss = (energy_pos - energy_neg).mean()

            im_loss = torch.pow(im_negs[:, -1:] - im[:, None], 2).mean()

            if it < 10000 or FLAGS.dataset != "celebahq_128":
                loss = im_loss
            else:
                vgg_loss = loss_fn_vgg(im_negs[:, -1], im).mean()
                loss = vgg_loss  + 0.1 * im_loss

            loss.backward()
            if FLAGS.gpus > 1:
                average_gradients(models)

            [torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) for model in models]
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad() for optimizer in optimizers]

            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()
                energy_pos_mean = energy_pos.mean().item()
                energy_neg_mean = energy_neg.mean().item()
                energy_pos_std = energy_pos.std().item()
                energy_neg_std = energy_neg.std().item()

                kvs = {}
                kvs['loss'] = loss
                kvs['ml_loss'] = ml_loss.item()
                kvs['im_loss'] = im_loss.item()

                if FLAGS.dataset == "celebahq_128" and ('vgg_loss' in kvs):
                    kvs['vgg_loss'] = vgg_loss.item()

                kvs['energy_pos_mean'] = energy_pos_mean
                kvs['energy_neg_mean'] = energy_neg_mean
                kvs['energy_pos_std'] = energy_pos_std
                kvs['energy_neg_std'] = energy_neg_std
                kvs['average_im_grad'] = torch.abs(im_grad).max()

                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                print(string)

            if it % FLAGS.save_interval == 0 and rank_idx == 0:
                model_path = osp.join(logdir, "model_{}.pth".format(it))


                ckpt = {'FLAGS': FLAGS}

                for i in range(len(models)):
                    ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()

                for i in range(len(optimizers)):
                    ckpt['optimizer_state_dict_{}'.format(i)] = optimizers[i].state_dict()

                torch.save(ckpt, model_path)
                print("Saving model in directory....")
                print('run test')

                test(test_dataloader, models, FLAGS, step=it)

            it += 1



def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus


    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except:
            pass

    if FLAGS.dataset == 'cubes':
        dataset = CubesColor(FLAGS, train=True)
        test_dataset = CubesColor(FLAGS, train=False)
    elif FLAGS.dataset == 'cubes_pair':
        dataset = CubesColorPair(FLAGS, train=True)
        test_dataset = CubesColorPair(FLAGS, train=False)
    elif FLAGS.dataset == "nvidia":
        dataset = Nvidia(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "clevr":
        dataset = Clevr(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "clevr_lighting":
        dataset = ClevrLighting(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "exercise":
        dataset = Exercise(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "intphys":
        dataset = IntPhysDataset(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "celebahq":
        dataset = CelebaHQ(resolution=64)
        test_dataset = dataset
    elif FLAGS.dataset == "celebahq_128":
        dataset = CelebaHQ(resolution=128)
        test_dataset = dataset
    elif FLAGS.dataset == "kitti":
        dataset = Kitti(FLAGS)
        test_dataset = dataset
    elif FLAGS.dataset == "faces":
        dataset = Faces(FLAGS)
        test_dataset = dataset
    else:
        dataset = ToyDataset(FLAGS)
        test_dataset = ToyDataset(FLAGS)

    shuffle=True
    sampler = None

    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:8113', world_size=world_size, rank=rank_idx, group_name="default")

    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}.pth".format(FLAGS.resume_iter))

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.num_visuals = FLAGS_OLD.num_visuals
        FLAGS.num_additional = FLAGS_OLD.num_additional
        FLAGS.decoder = FLAGS_OLD.decoder
        FLAGS.optimize_test = FLAGS_OLD.optimize_test
        FLAGS.temporal = FLAGS_OLD.temporal
        FLAGS.sim = FLAGS_OLD.sim
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.vae_beta = FLAGS_OLD.vae_beta

        models, optimizers  = init_model(FLAGS, device, dataset)
        state_dict = models[0].state_dict()

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict_{}'.format(i)], strict=False)

    else:
        models, optimizers = init_model(FLAGS, device, dataset)

    if FLAGS.gpus > 1:
        sync_model(models)

    if FLAGS.dataset == "multidsprites":
        train_dataloader = MultiDspritesLoader(FLAGS.batch_size)
        test_dataloader = MultiDspritesLoader(FLAGS.batch_size)
    elif FLAGS.dataset == "tetris":
        train_dataloader = TetrominoesLoader(FLAGS.batch_size)
        test_dataloader = TetrominoesLoader(FLAGS.batch_size)
    else:
        train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, pin_memory=False)
        test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.num_visuals, shuffle=True, pin_memory=False, drop_last=True)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    if FLAGS.train:
        models = [model.train() for model in models]
    else:
        models = [model.eval() for model in models]

    if FLAGS.train:
        train(train_dataloader, test_dataloader, logger, models, optimizers, FLAGS, logdir, rank_idx)

    elif FLAGS.optimize_test:
        test_optimize(test_dataloader, models, FLAGS, step=FLAGS.resume_iter)
    else:
        test(test_dataloader, models, FLAGS, step=FLAGS.resume_iter)


def main():
    FLAGS = parser.parse_args()
    FLAGS.ensembles = FLAGS.components
    FLAGS.tie_weight = True
    FLAGS.sample = True

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    main()
