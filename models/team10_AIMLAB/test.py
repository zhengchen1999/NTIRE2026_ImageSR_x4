import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from team10_SwinIR_Diff import SwinIR as net
from team10_SwinIR_Diff import SwinIRDenoiser


import time



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str,
                        default='../../model_zoo/team10_AIMLAB/team10_swinir_pretrained.pth')
    parser.add_argument('--folder_lq', type=str, default="/home/NTIRE26/data2/NTIRE2026/SR/dataset/DIV2K/Test/LR/X4", help='input low-quality test image folder')
    parser.add_argument('--diff_swinir_model_path', type=str, default='../../model_zoo/team10_AIMLAB/team10_swinir_diff.pth', help='Path to the best diff_swinir model')
    parser.add_argument('--output_dir', type=str, default='.result', help='Path to the best diff_swinir model')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    model = define_model(args)
    model.eval()
    model = model.to(device)

    denoised_model = SwinIRDenoiser(embed_dim=180, upscale=args.scale, in_chans=3)

    pretrained_model = torch.load(args.diff_swinir_model_path, map_location=device)
    denoised_model.load_state_dict(pretrained_model['model_state_dict'], strict=True)
    denoised_model = denoised_model.to(device)
    denoised_model.eval()

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    total_time = 0
    num_images = 0
    time_records = []


    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq = get_image_pair(args, path)  # image to HWC-BGR, float32
        # breakpoint()
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        start_time = time.time()

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            _, feature, mean, img_range, H, W = model(img_lq)
            output = denoised_model(feature, mean, img_range, H, W)

            output = output[..., :h_old * args.scale, :w_old * args.scale]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        single_time = end_time - start_time
        total_time += single_time
        num_images += 1
        time_records.append(single_time)


        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}.png', output)

        print(f"[{idx+1}] {imgname}: {single_time:.3f}s")
    
    if num_images > 0:
        avg_time = total_time / num_images
        max_time = max(time_records)
        min_time = min(time_records)
        
        print(f"\n========== Processing Summary ==========")
        print(f"Total images: {num_images}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"Average time: {avg_time:.3f}s per image")
        print(f"Fastest: {min_time:.3f}s")
        print(f"Slowest: {max_time:.3f}s")
        print(f"Throughput: {num_images/total_time:.2f} images/sec")
        print(f"========================================")



def define_model(args):

    model = net(upscale=args.scale, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'

    print(f'loading model from {args.model_path}')
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):

    save_dir = args.output_dir
    folder = args.folder_lq
    border = args.scale
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq


if __name__ == '__main__':
    main()
