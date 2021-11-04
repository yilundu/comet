from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from downsample import Downsample


def swish(x):
    return x * torch.sigmoid(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std*eps



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class CondResBlock(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64, latent_grid=False):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample
        self.latent_grid = latent_grid

        if filters <= 128:
            self.bn1 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.InstanceNorm2d(filters, affine=False)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=False)


        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=1e-5)

        # Upscale to an mask of image
        self.latent_fc1 = nn.Linear(latent_dim, 2*filters)
        self.latent_fc2 = nn.Linear(latent_dim, 2*filters)

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x, latent):
        x_orig = x

        latent_1 = self.latent_fc1(latent)
        latent_2 = self.latent_fc2(latent)

        gain = latent_1[:, :self.filters, None, None]
        bias = latent_1[:, self.filters:, None, None]

        gain2 = latent_2[:, :self.filters, None, None]
        bias2 = latent_2[:, self.filters:, None, None]

        x = self.conv1(x)
        x = gain * x + bias
        x = swish(x)


        x = self.conv2(x)
        x = gain2 * x + bias2
        x = swish(x)

        x_out = x_orig + x

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out


class CondResBlockNoLatent(nn.Module):
    def __init__(self, downsample=True, rescale=True, filters=64, upsample=False):
        super(CondResBlockNoLatent, self).__init__()

        self.filters = filters
        self.downsample = downsample

        if filters <= 128:
            self.bn1 = nn.GroupNorm(int(32  * filters / 128), filters, affine=True)
        else:
            self.bn1 = nn.GroupNorm(32, filters, affine=False)

        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        if filters <= 128:
            self.bn2 = nn.GroupNorm(int(32 * filters / 128), filters, affine=True)
        else:
            self.bn2 = nn.GroupNorm(32, filters, affine=True)

        self.upsample = upsample
        self.upsample_module = nn.Upsample(scale_factor=2)
        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

            self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        if upsample:
            self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_orig = x


        x = self.conv1(x)
        x = swish(x)

        x = self.conv2(x)
        x = swish(x)

        x_out = x_orig + x

        if self.upsample:
            x_out = self.upsample_module(x_out)
            x_out = swish(self.conv_downsample(x_out))

        if self.downsample:
            x_out = swish(self.conv_downsample(x_out))
            x_out = self.avg_pool(x_out)

        return x_out


class BroadcastConvDecoder(nn.Module):
    def __init__(self, im_size, latent_dim):
        super().__init__()
        self.im_size = im_size + 8
        self.latent_dim = latent_dim
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(self.latent_dim+2, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 0),
                    nn.ReLU(True),
                    nn.Conv2d(32, self.latent_dim, 1, 1, 0)
                    )

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)


    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = self.broadcast(z)
        x = self.g(z)
        return x


class LatentEBM128(nn.Module):
    def __init__(self, args, dataset):
        super(LatentEBM128, self).__init__()

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim

        self.components = args.components

        n_instance = len(dataset)
        self.pos_embed = args.pos_embed

        if self.pos_embed:
            self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv1_embed = nn.Conv2d(2, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, filter_dim // 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.gain = nn.Linear(args.latent_dim, filter_dim // 4)
        self.bias = nn.Linear(args.latent_dim, filter_dim // 4)

        self.recurrent_model = args.recurrent_model

        if args.dataset == "tetris":
            self.im_size = 35
        else:
            self.im_size = 64

        self.layer_encode = CondResBlock(filters=filter_dim//4, latent_dim=latent_dim, rescale=True)
        self.layer1 = CondResBlock(filters=filter_dim//2, latent_dim=latent_dim, rescale=True)
        self.layer2 = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer3 = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer4 = CondResBlock(filters=filter_dim, latent_dim=latent_dim)
        self.mask_decode = BroadcastConvDecoder(64, latent_dim)

        self.latent_map = nn.Linear(latent_dim, filter_dim * 8)
        self.energy_map = nn.Linear(filter_dim * 2, 1)

        self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)

        self.decode_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)

        self.latent_decode = nn.Conv2d(filter_dim, latent_dim_expand, kernel_size=3, stride=1, padding=1)

        self.downsample = Downsample(channels=args.latent_dim)
        self.dataset = args.dataset

        if self.recurrent_model:
            self.embed_layer4 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
            self.lstm = nn.LSTM(filter_dim, filter_dim, 1)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim)

            self.at_fc1 = nn.Linear(filter_dim*2, filter_dim)
            self.at_fc2 = nn.Linear(filter_dim, 1)

            self.map_embed = nn.Linear(filter_dim*2, filter_dim)

            if args.dataset == "tetris":
                self.pos_embedding = nn.Parameter(torch.zeros(9, filter_dim))
            else:
                self.pos_embedding = nn.Parameter(torch.zeros(16, filter_dim))
        else:
            self.embed_fc1 = nn.Linear(filter_dim, filter_dim)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim_expand)

        self.init_grid()

    def gen_mask(self, latent):
        return self.mask_decode(latent)

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

    def embed_latent(self, im):
        x = self.embed_conv1(im)
        x = F.relu(x)
        x = self.embed_layer1(x)
        x = self.embed_layer2(x)
        x = self.embed_layer3(x)

        if self.recurrent_model:

            #if self.dataset != "clevr":
            x = self.embed_layer4(x)

            s = x.size()
            x = x.view(s[0], s[1], -1)
            x = x.permute(0, 2, 1).contiguous()
            pos_embed = self.pos_embedding

            # x = x + pos_embed[None, :, :]
            h = torch.zeros(1, im.size(0), self.filter_dim).to(x.device), torch.zeros(1, im.size(0), self.filter_dim).to(x.device)
            outputs = []

            for i in range(self.components):
                (sx, cx) = h

                cx = cx.permute(1, 0, 2).contiguous()
                context = torch.cat([cx.expand(-1, x.size(1), -1), x], dim=-1)
                at_wt = self.at_fc2(F.relu(self.at_fc1(context)))
                at_wt = F.softmax(at_wt, dim=1)
                inp = (at_wt * context).sum(dim=1, keepdim=True)
                inp = self.map_embed(inp)
                inp = inp.permute(1, 0, 2).contiguous()

                output, h = self.lstm(inp, h)
                outputs.append(output)

            output = torch.cat(outputs, dim=0)
            output = output.permute(1, 0, 2).contiguous()
            output = self.embed_fc2(output)
            s = output.size()
            output = output.view(s[0], -1)
        else:
            x = x.mean(dim=2).mean(dim=2)

            x = x.view(x.size(0), -1)
            output = self.embed_fc1(x)
            x = F.relu(self.embed_fc1(x))
            output = self.embed_fc2(x)

        return output

    def forward(self, x, latent):

        if self.pos_embed:
            b = x.size(0)
            x_grid = self.x_grid.expand(b, 1, -1, -1).to(x.device)
            y_grid = self.y_grid.expand(b, 1, -1, -1).to(x.device)
            coord_grid = torch.cat([x_grid, y_grid], dim=1)

        inter = self.conv1(x)
        inter = swish(inter)

        if self.pos_embed:
            pos_inter = self.conv1_embed(coord_grid)
            pos_inter = swish(pos_inter)

            inter = torch.cat([inter, pos_inter], dim=1)


        x = self.layer_encode(inter, latent)
        x = self.layer1(x, latent)


        x = self.layer2(x, latent)
        x = self.layer3(x, latent)
        x = self.layer4(x, latent)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)

        energy = self.energy_map(x)

        return energy


class LatentEBM(nn.Module):
    def __init__(self, args, dataset):
        super(LatentEBM, self).__init__()

        filter_dim = args.filter_dim
        self.filter_dim = filter_dim
        latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim

        self.components = args.components

        n_instance = len(dataset)
        self.pos_embed = args.pos_embed

        if self.pos_embed:
            self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
            self.conv1_embed = nn.Conv2d(2, filter_dim // 2, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)

        self.gain = nn.Linear(args.latent_dim, filter_dim)
        self.bias = nn.Linear(args.latent_dim, filter_dim)

        self.recurrent_model = args.recurrent_model

        if args.dataset == "tetris":
            self.im_size = 35
        else:
            self.im_size = 64

        self.layer_encode = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer1 = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer2 = CondResBlock(filters=filter_dim, latent_dim=latent_dim)
        self.mask_decode = BroadcastConvDecoder(64, latent_dim)

        self.latent_map = nn.Linear(latent_dim, filter_dim * 8)
        self.energy_map = nn.Linear(filter_dim * 2, 1)

        self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
        self.embed_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)

        self.decode_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer2 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)
        self.decode_layer3 = CondResBlockNoLatent(filters=filter_dim, rescale=False, upsample=True, downsample=False)

        self.latent_decode = nn.Conv2d(filter_dim, latent_dim_expand, kernel_size=3, stride=1, padding=1)

        self.downsample = Downsample(channels=args.latent_dim)
        self.dataset = args.dataset

        if self.recurrent_model:
            self.embed_layer4 = CondResBlockNoLatent(filters=filter_dim, rescale=False, downsample=True)
            self.lstm = nn.LSTM(filter_dim, filter_dim, 1)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim)

            self.at_fc1 = nn.Linear(filter_dim*2, filter_dim)
            self.at_fc2 = nn.Linear(filter_dim, 1)

            self.map_embed = nn.Linear(filter_dim*2, filter_dim)

            if args.dataset == "tetris":
                self.pos_embedding = nn.Parameter(torch.zeros(9, filter_dim))
            else:
                self.pos_embedding = nn.Parameter(torch.zeros(16, filter_dim))
        else:
            self.embed_fc1 = nn.Linear(filter_dim, filter_dim)
            self.embed_fc2 = nn.Linear(filter_dim, latent_dim_expand)

        self.init_grid()

    def gen_mask(self, latent):
        return self.mask_decode(latent)

    def init_grid(self):
        x = torch.linspace(0, 1, self.im_size)
        y = torch.linspace(0, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

    def embed_latent(self, im):
        x = self.embed_conv1(im)
        x = F.relu(x)
        x = self.embed_layer1(x)
        x = self.embed_layer2(x)
        x = self.embed_layer3(x)

        if self.recurrent_model:

            x = self.embed_layer4(x)

            s = x.size()
            x = x.view(s[0], s[1], -1)
            x = x.permute(0, 2, 1).contiguous()

            h = torch.zeros(1, im.size(0), self.filter_dim).to(x.device), torch.zeros(1, im.size(0), self.filter_dim).to(x.device)
            outputs = []

            for i in range(self.components):
                (sx, cx) = h

                cx = cx.permute(1, 0, 2).contiguous()
                context = torch.cat([cx.expand(-1, x.size(1), -1), x], dim=-1)
                at_wt = self.at_fc2(F.relu(self.at_fc1(context)))
                at_wt = F.softmax(at_wt, dim=1)
                inp = (at_wt * context).sum(dim=1, keepdim=True)
                inp = self.map_embed(inp)
                inp = inp.permute(1, 0, 2).contiguous()

                output, h = self.lstm(inp, h)
                outputs.append(output)

            output = torch.cat(outputs, dim=0)
            output = output.permute(1, 0, 2).contiguous()
            output = self.embed_fc2(output)
            s = output.size()
            output = output.view(s[0], -1)
        else:
            x = x.mean(dim=2).mean(dim=2)

            x = x.view(x.size(0), -1)
            output = self.embed_fc1(x)
            x = F.relu(self.embed_fc1(x))
            output = self.embed_fc2(x)

        return output

    def forward(self, x, latent):

        if self.pos_embed:
            b = x.size(0)
            x_grid = self.x_grid.expand(b, 1, -1, -1).to(x.device)
            y_grid = self.y_grid.expand(b, 1, -1, -1).to(x.device)
            coord_grid = torch.cat([x_grid, y_grid], dim=1)

        # x = x.contiguous()
        inter = self.conv1(x)
        inter = swish(inter)

        if self.pos_embed:
            pos_inter = self.conv1_embed(coord_grid)
            pos_inter = swish(pos_inter)

            inter = torch.cat([inter, pos_inter], dim=1)

        x = self.avg_pool(inter)

        x = self.layer_encode(x, latent)

        x = self.layer1(x, latent)
        x = self.layer2(x, latent)

        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)

        energy = self.energy_map(x)

        return energy

class ToyEBM(nn.Module):
    def __init__(self, args, dataset):
        super(ToyEBM, self).__init__()

        filter_dim = args.filter_dim
        latent_dim_expand = args.latent_dim * args.components
        latent_dim = args.latent_dim
        im_size = args.im_size

        n_instance = len(dataset)

        self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.layer_encode = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer1 = CondResBlock(filters=filter_dim, latent_dim=latent_dim, rescale=False)
        self.layer2 = CondResBlock(filters=filter_dim, latent_dim=latent_dim)


        self.fc1 = nn.Linear(filter_dim * 2, filter_dim * 2)

        self.latent_map = nn.Linear(latent_dim, filter_dim * 8)
        self.energy_map = nn.Linear(filter_dim * 2, 1)

        self.embed_conv1 = nn.Conv2d(3, filter_dim, kernel_size=5, stride=2, padding=3)
        self.embed_layer1 = CondResBlockNoLatent(filters=filter_dim, rescale=False)
        self.embed_layer2 = CondResBlockNoLatent(filters=filter_dim)
        self.embed_fc1 = nn.Linear(filter_dim * 2, filter_dim * 2)
        self.embed_fc2 = nn.Linear(filter_dim * 2, latent_dim_expand)

        self.steps = torch.nn.parameter.Parameter(torch.ones(args.num_steps), requires_grad=True)


    def embed_latent(self, im):
        x = self.embed_conv1(im)
        x = F.relu(x)
        x = self.embed_layer1(x)
        x = self.embed_layer2(x)
        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.embed_fc1(x))
        output = self.embed_fc2(x)

        return output

    def forward(self, x, latent):
        x = swish(self.conv1(x))
        x = self.avg_pool(x)
        x = self.layer_encode(x, latent)
        x = self.layer1(x, latent)
        x = self.layer2(x, latent)
        x = x.mean(dim=2).mean(dim=2)
        x = x.view(x.size(0), -1)


        x = swish(self.fc1(x))
        energy = self.energy_map(x)

        return energy


class DisentangleModel(nn.Module):
    def __init__(self):
        super(DisentangleModel, self).__init__()

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class PMM(nn.Module):
    def __init__(self, args):
        super(PMM, self).__init__()

        latent_dim = args.latent_dim * args.components
        self.latent_dim = latent_dim
        self.inner_dim = 1024

        self.fc1 = nn.Linear(self.latent_dim, self.inner_dim)
        self.fc2 = nn.Linear(self.inner_dim, self.inner_dim)
        self.fc3 = nn.Linear(self.inner_dim, self.inner_dim)
        self.fc4 = nn.Linear(self.inner_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))

        return x


class LabelModel(nn.Module):
    def __init__(self, args):
        super(LabelModel, self).__init__()

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)

        return x

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def reconstruction_loss(self, x, x_recon, distribution):
        batch_size = x.size(0)
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        else:
            recon_loss = None

        return recon_loss

    def compute_cross_ent_normal(self, mu, logvar):
        return 0.5 * (mu**2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

    def compute_ent_normal(self, logvar):
        return 0.5 * (logvar + np.log(2 * np.pi * np.e))


if __name__ == "__main__":
    args = EasyDict()
    args.filter_dim = 64
    args.latent_dim = 64
    args.im_size = 256

    model = LatentEBM(args).cuda()
    x = torch.zeros(1, 3, 256, 256).cuda()
    latent = torch.zeros(1, 64).cuda()
    model(x, latent)
