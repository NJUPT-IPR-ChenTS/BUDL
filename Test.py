import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from models.BUDL_model import *
from SIVPv1 import *

transform1 = transforms.Compose([
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def loader_Image(path):
    '''
    path:图片路径
    函数说明：读取图片，并按照训练时的方式转换到tensor变量形式
    '''
    img = Image.open(path)  #加载图片为PIL格式
    #img_shape = np.array(img.size)  #获取图片格式
    img = img.convert('RGB')  #将图片转为RGB

    img_tensor = transform1(img)
    dark = img_tensor  # 输入暗图像
    R_split, G_split, B_split = torch.split(dark, 1, dim=0)
    zero_array = R_split * G_split * B_split
    zero_array[zero_array != 0] = 1
    zero_array = 1 - zero_array
    mask = zero_array

    img_tensor = transform2(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    mask = mask.unsqueeze(0)
    return img_tensor, mask

def auto_padding(img, times=32):
    # img: numpy image with shape H*W*C

    b,c,h,w = img.shape
    h_pad = times - h % times if not h % times == 0 else 0
    w_pad = times - w % times if not w % times == 0 else 0
    img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')
    return img

def test_on_images(floder_path, epoch):
    '''
    floder_path:测试图片所在的文件夹地址
    save_path:保存生成图片文件夹的地址
    generator_path：生成器保存的模型参数
    '''
    import time
    # 获取当前时间戳
    timestamp1 = time.time()  # 返回浮点数，表示自1970年1月1日以来的秒数

    a = epoch
    for e in np.arange(a, a+5, 5):

        os.makedirs(r'test\xuezhang\model_{}'.format(e), exist_ok=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cycle_G_AB = BUDL(upscale=2, img_size=(256, 256),
                   Tran_embed_dim=32, img_range=1., num_blocks=[1, 1, 1, 1],
                   embed_dim=64, resi_connection='dbconv',upsampler='pixelshuffledirect', trainning=False).to(device)
        cycle_G_BA = BUDL_1(upscale=2, img_size=(256, 256),
                   Tran_embed_dim=32, img_range=1., num_blocks=[1, 1, 1, 1],
                   embed_dim=64, resi_connection='dbconv',upsampler='pixelshuffledirect', trainning=False).to(device)

        cycle_G_AB.load_state_dict(torch.load(r"D:\chen\SwinIR_CycleGAN2\saved_models\move_loss_BDSP\G_AB_{}.pth".format(e)))
        cycle_G_AB.eval()
        cycle_G_BA.load_state_dict(torch.load(r"D:\chen\SwinIR_CycleGAN2\saved_models\move_loss_BDSP\G_BA_{}.pth".format(e)))
        cycle_G_BA.eval()

        #加载测试图片
        for ImgName in os.listdir(floder_path):
            temp_img_path = os.path.join(floder_path, ImgName)  #图片路径
            img_var, zero=loader_Image(temp_img_path)
            cycle_G_BA = cycle_G_BA.cuda()  # Move model to GPU
            img_var = img_var.cuda()  # Ensure input is on GPU
            zero = zero.to(device)

            with torch.no_grad():
                gen_img= cycle_G_BA(img_var,zero)
                gen_img = (gen_img +1) /2
                save_image(gen_img, os.path.join(r'D:\chen\\results\LOLv2-pc\our_cycle_enhance_ntm65', ImgName), normalize=False)
        timestamp2 = time.time()

        # 时间戳相减
        difference = timestamp2 - timestamp1
        print(f"时间差: {difference} 秒")  # 约2.0秒
        print("epoch {} have done!".format(e))

testImage_path = r'D:\chen\NTM_5000\A'
test_on_images(testImage_path, epoch=100)


