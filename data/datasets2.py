
import random
import torch.utils.data as data
import torch

from utils import utils_image as util
import torchvision.transforms as transforms
import os
from tqdm import tqdm

def store_all_data(paths_H, paths_L):
    img_h = []
    img_l = []
    for i in tqdm(range(len(paths_H))):
        H_path = paths_H[i]
        img_H = util.imread_uint(H_path, 3)
        img_h.append(img_H)

    for i in tqdm(range(len(paths_L))):
        L_path = paths_L[i]
        img_L = util.imread_uint(L_path, 3)
        img_l.append(img_L)

    return img_h, img_l

class ImageDataset(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, root, size,transforms_=None, unaligned=False, mode="train"):
        super(ImageDataset, self).__init__()
        print('Dataset: Enlighten on AWGN with fixed sigma. Only dataroot_H And dataroot_L are needed!')

        self.n_channels = 3
        self.patch_size = size
        self.sigma = 25
        self.sigma_test = self.sigma
        self.mode = mode
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = sorted(util.get_image_paths(os.path.join(root, "%s/B" % mode)))
        self.paths_L = sorted(util.get_image_paths(os.path.join(root, "%s/A" % mode)))

        self.transform = transforms.Compose(
            [#  transforms.ToTensor(),
                # transforms.Resize((800,1048)),
                # transforms.RandomCrop(320),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])
        self.transform2 = transforms.Compose(
            [#  transforms.ToTensor(),
                #transforms.CenterCrop(320),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))])
        self.img_H, self.img_L = store_all_data(self.paths_H, self.paths_L)
        self.H_size = len(self.img_H)
        self.L_size = len(self.img_L)

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------

        img_H = self.img_H[index % self.H_size]

        img_L = self.img_L[index % self.L_size]

        if self.mode == 'train':
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)
            patch_L = util.augment_img(patch_L, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(patch_L)
            dark = img_H  # 输入暗图像
            R_split, G_split, B_split = torch.split(dark, 1, dim=0)
            zero_array = R_split * G_split * B_split
            zero_array[zero_array != 0] = 1
            zero_array = 1 - zero_array
            mask = zero_array
            img_H = self.transform(img_H)
            img_L = self.transform(img_L)

            # num = mask.sum()
        else:

            img_H = util.uint2single(img_H)
            img_L = util.uint2single(img_L)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)
            dark = img_H  # 输入暗图像
            R_split, G_split, B_split = torch.split(dark, 1, dim=0)
            zero_array = R_split * G_split * B_split
            zero_array[zero_array != 0] = 1
            zero_array = 1 - zero_array
            mask = zero_array
            img_H = self.transform2(img_H)
            img_L = self.transform2(img_L)


        return {"A": img_L, "B": img_H, "zero": mask}

    def __len__(self):
        return max(self.H_size, self.L_size)
