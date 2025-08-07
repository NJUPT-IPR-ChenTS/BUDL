import os
import glob
import argparse
from torchvision import utils
# from basicsr.utils.options import yaml_load
from MMD.gfpgan.data.ffhq_degradation_specific import FFHQDegradationSpecific
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ffhq_cropface_512_test
# ffhq_cropface_dg512_train
# ffhq_cropface_dg512_test
# data_augment_origin
def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str,required=False, help='Path to option YAML file.',default=r'.\MMD\options\data_augment.yml')
    parser.add_argument('-i','--input',type=str,default=r'D:\chen\NTM\11')
    parser.add_argument('-o', '--output', type=str, default=r'D:\chen\Data_augmet\aug_results', help='Output folder. Default: results')
    args = parser.parse_args()
    # opt = yaml_load(args.opt)
    import yaml

    with open(r'.\MMD\options\data_augment.yml', 'r', encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up degradation ------------------------
    degradater = FFHQDegradationSpecific(opt['datasets']['train'])

    # ------------------------ degrade ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        pre_name = os.path.splitext(img_name)[0]
        # pre_name = int(pre_name) + 70000
        print(f'Processing {img_name} ...')
        img_lr = degradater.img_deg(img_path)
        utils.save_image(
            img_lr,
            # f'D:/Datasets/CFF_final/{str(pre_name).zfill(5)}.png',
            # f'D:\\chen\\NTM\B\\{str(pre_name).zfill(5)}.jpg',
            os.path.join(args.output, f'{str(pre_name).zfill(5)}.jpg'),
            normalize=False,
            range=(-1, 1))
        # for i in range(len(inter_img)):
        #     utils.save_image(
        #         inter_img[i],
        #         f'datasets/visual_severe_deg/skips{str(i)}/{str(img_count).zfill(6)}.png',
        #         normalize=True,
        #         range=(-1, 1),
        #     )

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
