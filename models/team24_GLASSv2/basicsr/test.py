import logging
import torch
import os
from os import path as osp
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    '''for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])'''
    total_time = 0.0
    total_images = 0

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        
        # Start timer for forward passes
        start_time = time.time()
        
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        
        # Stop timer
        end_time = time.time()
        
        total_time += (end_time - start_time)
        total_images += len(test_loader.dataset)

    # Automatically generate readme.txt after all images are processed
    if total_images > 0:
        runtime_per_img = total_time / total_images
        
        readme_content = (
            f"runtime per image [s] : {runtime_per_img:.4f}\n"
            f"CPU[1] / GPU[0] : 0\n"
            f"Extra Data [1] / No Extra Data [0] : 1\n"
            f"Other description : Solution based on GLASS (Super Resolution x4). We utilize a PyTorch implementation running on a single GPU. The method was trained on the standard DIV2K and Flickr2K datasets."
        )
        
        # Save the readme.txt in the root path
        readme_path = osp.join(root_path, 'readme.txt')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        logger.info(f"Successfully generated Codabench readme at: {readme_path}")
        print("\n--- Generated readme.txt content ---")
        print(readme_content)
        print("------------------------------------\n")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
