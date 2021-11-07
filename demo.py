import argparse
from models import LatentEBM128
from imageio import imread, get_writer
from skimage.transform import resize as imresize
import torch


def gen_image(latents, FLAGS, models, im_neg, num_steps, idx=None):
    im_negs = []

    im_neg.requires_grad_(requires_grad=True)

    for i in range(num_steps):
        energy = 0

        for j in range(len(latents)):
            if idx is not None and idx != j:
                pass
            else:
                ix = j % FLAGS.components
                energy = models[j % FLAGS.components].forward(im_neg, latents[j]) + energy

        im_grad, = torch.autograd.grad([energy.sum()], [im_neg])

        im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = torch.clamp(im_neg, 0, 1)
        im_negs.append(im_neg)
        im_neg = im_neg.detach()
        im_neg.requires_grad_()

    return im_negs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train EBM model')
    parser.add_argument('--im_path', default='im_0.jpg', type=str, help='image to load')
    args = parser.parse_args()

    ckpt = torch.load("celebahq_128.pth")
    FLAGS = ckpt['FLAGS']
    state_dict = ckpt['model_state_dict_0']

    model = LatentEBM128(FLAGS, 'celebahq_128').cuda()
    model.load_state_dict(state_dict)
    models = [model for i in range(4)]


    im = imread(args.im_path)
    im = imresize(im, (128, 128))
    im = torch.Tensor(im).permute(2, 0, 1)[None, :, :, :].contiguous().cuda()

    latent = model.embed_latent(im)
    latents = torch.chunk(latent, 4, dim=1)

    im_neg = torch.rand_like(im)

    FLAGS.step_lr = 200.0
    ims = gen_image(latents, FLAGS, models, im_neg, 30)

    writer = get_writer("im_opt_full.mp4")
    for im in ims:
        im = im.detach().cpu().numpy()[0]
        im = im.transpose((1, 2, 0))
        writer.append_data(im)

    writer.close()

    for i in range(4):
        writer = get_writer("im_opt_{}.mp4".format(i))

        ims = gen_image(latents, FLAGS, models, im_neg, 30, idx=i)

        for im in ims:
            im = im.detach().cpu().numpy()[0]
            im = im.transpose((1, 2, 0))
            writer.append_data(im)

        writer.close()
