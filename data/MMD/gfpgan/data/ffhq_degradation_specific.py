import cv2
import math
import numpy as np
# import os.path as osp
import torch
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)


class FFHQDegradationSpecific():
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(FFHQDegradationSpecific, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        self.out_size = opt['out_size']

        self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1)  # whether enlarge eye regions

        # file client (lmdb io backend)
        #   从opt['dataroot_gt']调用数据列表

        #   disk应该是磁盘中读取
        # disk backend: scan file list from a folder
        #   对文件以及相对路径进行合成
        self.paths = paths_from_folder(self.gt_folder)

        # degradation configurations
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.downsample_range = opt['downsample_range']
        self.noise_range = opt['noise_range']
        self.jpeg_range = opt['jpeg_range']

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob')
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob')
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        # to gray
        self.gray_prob = opt.get('gray_prob')

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    # def brighter(self, img, percetage=1.07):
    #     """增强图片像素，使图片变亮"""
    #     image_brighter = img.copy()
    #     w = img.shape[1]
    #     h = img.shape[0]
    #     # get brighter
    #     for xi in range(0, w):
    #         for xj in range(0, h):
    #             image_brighter[xj, xi, 0] = np.clip(int(img[xj, xi, 0] * percetage), a_max=255, a_min=0)
    #             image_brighter[xj, xi, 1] = np.clip(int(img[xj, xi, 1] * percetage), a_max=255, a_min=0)
    #             image_brighter[xj, xi, 2] = np.clip(int(img[xj, xi, 2] * percetage), a_max=255, a_min=0)
    #     percetage_name = str(percetage * 100).replace('.', '')
    #     operate_name = 'brighter_' + percetage_name
    #     save_name = self.get_savename(operate_name)
    #     cv2.imwrite(save_name, image_brighter)
    #     print('{}做数据增强{} 完毕 '.format(self.class_name, operate_name))

    def img_deg(self, img_path):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        img_bytes = self.file_client.get(img_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        # 使用自定义内核对图像进行卷积
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # 生成均匀分布的随机数
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.opt.get('gt_gray'):  # whether convert GT to gray images
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])  # repeat the color channels

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # img_lq = img2tensor(img_gt, bgr2rgb=True, float32=True)
        #
        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        # normalize(img_gt, self.mean, self.std, inplace=True)
        # normalize(img_lq, self.mean, self.std, inplace=True)

        return img_lq

