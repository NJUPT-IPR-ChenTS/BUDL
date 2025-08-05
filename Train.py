import argparse
import itertools

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from SIVPv2 import *
from SIVPv0 import *
from SIVPv1 import *
from datasets2 import *
from utils import *
import lossfn
import torch.nn as nn
import torch.nn.functional as F
import torch
from BDSP_Face import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2020)

# cycleGAN+BDSP_1
parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name", type=str, default="NOFOURLIA_Diffusion36", help="name of the experiment")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default=r"E:\TAO\Dataset\NTM_5000", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=384, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=600, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=5.0, help="cycle loss weight")#5
parser.add_argument("--lambda_id", type=float, default=1.0, help="identity loss weight")
parser.add_argument("--lambda_dsp", type=float, default=10.0, help="dsp loss weight")#10

opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.experiment_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.experiment_name, exist_ok=True)
os.makedirs("test/%s" % opt.experiment_name, exist_ok=True)
# Losses
criterion_GAN = torch.nn.MSELoss()  # 该统计参数是预测数据和原始数据对应点误差的平方和的均值
criterion_cycle = torch.nn.L1Loss()  # 它是把目标值  与模型输出（估计值）做绝对值得到的误差。
criterion_identity = torch.nn.L1Loss()  # 它是把目标值 与模型输出（估计值）  做绝对值得到的误差。
criterion_BDSP = torch.nn.L1Loss()  # 它是把目标值 与模型输出（估计值）  做绝对值得到的误差。
vgg = lossfn.load_vgg16("./model")
vgg.eval()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)
device = torch.device('cuda')
# Initialize generator and discriminator
G_AB = BUDL(upscale=1, in_chans=3, img_size=256, window_size=8,
            img_range=1., depths=[2, 2, 2, 2], embed_dim=64, num_heads=[6, 6, 6, 6],
            mlp_ratio=2, upsampler='null', resi_connection='1conv', train=True)
G_BA = BUDL_1(upscale=1, in_chans=3, img_size=256, window_size=8,
            img_range=1., depths=[2, 2, 2, 2], embed_dim=64, num_heads=[6, 6, 6, 6],
            mlp_ratio=2, upsampler='null', resi_connection='1conv', train=True)

print(G_AB)
D_A = Discriminator(5)
dis_A_P = Discriminator(4)
D_B = Discriminator(5)
dis_B_P = Discriminator(4)
print(D_A)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    dis_A_P = dis_A_P.cuda()
    dis_B_P = dis_B_P.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    vgg.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\G_AB_180.pth"))
    G_BA.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\G_BA_180.pth"))
    D_A.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\D_A_180.pth"))
    D_B.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\D_B_180.pth"))
else:
    #    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A_p = torch.optim.Adam(dis_A_P.parameters(),lr=0.0001,betas=(0.5,0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B_p = torch.optim.Adam(dis_B_P.parameters(),lr=0.0001,betas=(0.5,0.999))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A_p = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A_p, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

lr_scheduler_D_B_p = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B_p, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_A_p_buffer = ReplayBuffer()
fake_B_p_buffer = ReplayBuffer()
# Image transformations
transforms_ = [
    #transforms.Resize(320),
    #transforms.RandomCrop(320),
    #transforms.RandomHorizontalFlip(0.5),
    #transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms_2 = [
    #transforms.Resize(int(opt.img_height), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
# "data/%s" % opt.dataset_name
dataloader = DataLoader(
    ImageDataset('%s' % opt.dataset_name,opt.img_height,transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset('%s' % opt.dataset_name, opt.img_height,transforms_=transforms_2, unaligned=False, mode="test"),
    batch_size=3,
    shuffle=True,
    num_workers=0,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A,zero)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B,zero)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.experiment_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
# 训练批次循环
for epoch in range(opt.epoch, opt.n_epochs):
    # 对train集数据加载，获得其索引和图片

    for j, batch in enumerate(dataloader):


        real_A_patch = []
        real_B_patch = []
        fake_A_patch = []
        fake_B_patch = []
        # Set model input
        # A和B分别是油画和真实图片
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        zero = Variable(batch["zero"].type(Tensor))
        num11=zero.sum()
        w = real_A.size(3)
        h = real_A.size(2)
        # Adversarial ground truths
        # valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        # 这两个是生成器，此处是训练模式
        G_AB.train()
        G_BA.train()

        # 经典四步走
        optimizer_G.zero_grad()

        # id loss
        loss_id_A = criterion_identity(G_BA(real_A,zero), real_A)  # 身份损失是生成图片与真实图片之间的L1Loss()
        loss_id_B = criterion_identity(G_AB(real_B,zero), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A,zero)
        D_pred_fakeB = D_B(fake_B)
        D_pred_realB = D_B(real_B)
        loss_GAN_AB = (criterion_GAN(D_pred_realB - torch.mean(D_pred_fakeB), torch.zeros_like(D_pred_realB)) +
                       criterion_GAN(D_pred_fakeB - torch.mean(D_pred_realB), torch.ones_like(D_pred_fakeB))) / 2

        fake_A = G_BA(real_B,zero)
        D_pred_fakeA = D_A(fake_A)
        D_pred_realA = D_A(real_A)
        loss_GAN_BA = (criterion_GAN(D_pred_realA - torch.mean(D_pred_fakeA), torch.zeros_like(D_pred_realA)) +
                       criterion_GAN(D_pred_fakeA - torch.mean(D_pred_realA), torch.ones_like(D_pred_fakeA))) / 2

        for i in range(5):
            offset1 = random.randint(0, max(0, h - 64 - 1))
            offset2 = random.randint(0, max(0, w - 64 - 1))

            real_A_patch.append(real_A[:, :, offset1:offset1 + 64, offset2:offset2 + 64])
            real_B_patch.append(real_B[:, :, offset1:offset1 + 64, offset2:offset2 + 64])
            fake_A_patch.append(fake_A[:, :, offset1:offset1 + 64, offset2:offset2 + 64])
            fake_B_patch.append(fake_B[:, :, offset1:offset1 + 64, offset2:offset2 + 64])

        gen_loss_BA_p = 0
        gen_loss_AB_p = 0
        loss_vgg_a_p = 0
        loss_vgg_b_p = 0
        for i in range(5):
            D_pred_fakeA_p = dis_A_P(fake_A_patch[i])
            gen_loss_AB_p += criterion_GAN(D_pred_fakeA_p,torch.ones_like(D_pred_fakeA_p))

            D_pred_fakeB_p = dis_B_P(fake_B_patch[i])
            gen_loss_BA_p += criterion_GAN(D_pred_fakeB_p,torch.ones_like(D_pred_fakeB_p))

            loss_vgg_a_p += lossfn.compute_vgg_loss(vgg, fake_A_patch[i], real_B_patch[i]).to(device)
            loss_vgg_b_p += lossfn.compute_vgg_loss(vgg, fake_B_patch[i], real_A_patch[i]).to(device)

        loss_GAN_AB += gen_loss_AB_p / 5
        loss_GAN_BA += gen_loss_BA_p / 5
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        loss_vgg_a = lossfn.compute_vgg_loss(vgg, fake_A, real_B)
        loss_vgg_b = lossfn.compute_vgg_loss(vgg, fake_B, real_A)
        loss_vgg_a += loss_vgg_a_p/5
        loss_vgg_b += loss_vgg_b_p/5
        loss_vgg = (loss_vgg_a + loss_vgg_b) / 2

        loss_BDSP_A = criterion_BDSP(BDSP_Face(fake_B), BDSP_Face(real_A))
        loss_BDSP_B = criterion_BDSP(BDSP_Face(fake_A), BDSP_Face(real_B))
        loss_BDSP = (loss_BDSP_A + loss_BDSP_B) / 2

        # Cycle loss
        recov_A = G_BA(fake_B,zero)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A,zero)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss_BDSP_RA = criterion_BDSP(BDSP_Face(recov_A), BDSP_Face(real_A))
        loss_BDSP_RB = criterion_BDSP(BDSP_Face(recov_B), BDSP_Face(real_B))
        loss_BDSP_R = (loss_BDSP_RA + loss_BDSP_RB) / 2

        # Total loss
        loss_G = loss_GAN + loss_cycle *2 + loss_identity + loss_vgg /2 + opt.lambda_dsp*loss_BDSP + opt.lambda_dsp*loss_BDSP_R

        loss_G.backward()
        optimizer_G.step()

        # 训练判别器A
        optimizer_D_A.zero_grad()
        D_pred_real_A = D_A(real_A)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        D_pred_fake_A = D_A(fake_A_.detach())
        loss_D_A = (criterion_GAN(D_pred_real_A - torch.mean(D_pred_fake_A), torch.ones_like(D_pred_real_A)) +
                    criterion_GAN(D_pred_fake_A - torch.mean(D_pred_real_A), torch.zeros_like(D_pred_fake_A))) / 2
        loss_D_A.backward()
        optimizer_D_A.step()


        # 训练判别器B
        optimizer_D_B.zero_grad()
        D_pred_real_B = D_B(real_B)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        D_pred_fake_B = D_B(fake_B_.detach())
        loss_D_B = (criterion_GAN(D_pred_real_B - torch.mean(D_pred_fake_B), torch.ones_like(D_pred_real_B)) +
                    criterion_GAN(D_pred_fake_B - torch.mean(D_pred_real_B), torch.zeros_like(D_pred_fake_B))) / 2
        loss_D_B.backward()
        optimizer_D_B.step()


        dis_A_loss_p = 0
        optimizer_D_A_p.zero_grad()
        # pred_real_A_p = dis_A_P(real_A_patch[2])
        # pred_fake_A_p = dis_A_P(fake_A_patch[2].detach())
        # dis_A_loss_pp =(criterion_GAN(pred_real_A_p, torch.ones_like(pred_real_A_p)) + criterion_GAN(pred_fake_A_p, torch.zeros_like(pred_fake_A_p))) / 2
        for i in range(5):
            pred_real_A_p = dis_A_P(real_A_patch[i])
            pred_fake_A_p = dis_A_P(fake_A_p_buffer.push_and_pop(fake_A_patch[i]).detach())
            dis_A_loss_p += (criterion_GAN(pred_real_A_p,  torch.ones_like(pred_real_A_p)) +
                             criterion_GAN(pred_fake_A_p, torch.zeros_like(pred_fake_A_p))) /2
        dis_A_loss_pp = dis_A_loss_p/5
        dis_A_loss_pp.backward()
        optimizer_D_A_p.step()

        dis_B_loss_p = 0
        optimizer_D_B_p.zero_grad()
        # pred_real_B_p = dis_A_P(real_B_patch[2])
        # pred_fake_B_p = dis_A_P(fake_B_patch[2].detach())
        # dis_B_loss_pp =(criterion_GAN(pred_real_B_p, torch.ones_like(pred_real_B_p)) + criterion_GAN(pred_fake_B_p, torch.zeros_like(pred_fake_B_p))) / 2
        for i in range(5):
            pred_real_B_p = dis_B_P(real_B_patch[i])
            pred_fake_B_p = dis_B_P(fake_B_p_buffer.push_and_pop(fake_B_patch[i]).detach())
            dis_B_loss_p += (criterion_GAN(pred_real_B_p,  torch.ones_like(pred_real_B_p)) +
                             criterion_GAN(pred_fake_B_p, torch.zeros_like(pred_fake_B_p))) /2
        dis_B_loss_pp = dis_B_loss_p/5
        dis_B_loss_pp.backward()
        optimizer_D_B_p.step()

        loss_D = (loss_D_A + loss_D_B + dis_A_loss_pp + dis_B_loss_pp )

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + j
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f,loss_vgg: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                j,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                loss_vgg.item(),
                time_left,
            )
        )

        # If at sample interval save image
        # if batches_done % opt.sample_interval == 0:
        #     sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    lr_scheduler_D_A_p.step()
    lr_scheduler_D_B_p.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch >= 60:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.experiment_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.experiment_name, epoch))
        #torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.experiment_name, epoch))
        #torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.experiment_name, epoch))

# torch.save(G_AB.state_dict(), "saved_models/%s/%s/G_AB_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
# torch.save(G_BA.state_dict(), "saved_models/%s/%s/G_BA_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
# torch.save(D_A.state_dict(), "saved_models/%s/%s/D_A_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
# torch.save(D_B.state_dict(), "saved_models/%s/%s/D_B_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))

