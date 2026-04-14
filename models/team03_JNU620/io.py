import os
import logging
import torch
import time
from os import path as osp

from basicsr.utils import yaml_load, get_root_logger, tensor2img, imwrite
# from basicsr.utils import get_root_logger, tensor2img, imwrite
# import yaml
# def yaml_load(f): return yaml.safe_load(open(f, "r").read())
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model


def main(model_dir, input_path, output_path, device=None):
    logger = get_root_logger(logger_name='team03_ProCata', log_level=logging.INFO)

    # load option yaml
    opt_file = osp.join(osp.dirname(__file__), 'options', 'test', 'test_ProCata_x4.yml')
    if not osp.isfile(opt_file):
        raise FileNotFoundError(f"Option file not found: {opt_file}")
    opt = yaml_load(opt_file)

    # Minimal post-processing of options (emulate parse_options behavior)
    opt.setdefault('num_gpu', 1)
    opt.setdefault('dist', False)
    opt.setdefault('manual_seed', 3407)
    # ensure datasets have 'phase' and expanded paths
    for phase_key, dataset in opt.get('datasets', {}).items():
        phase = phase_key.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # override paths to point to provided input_path / output_path
    # assume input_path contains subfolders 'LQ' and optionally 'HR'
    # lq_folder = osp.join(input_path, 'LQ')
    lq_folder = input_path
    gt_folder = osp.join(input_path, 'HR')

    # update dataset roots for all datasets in the yaml
    for key, dataset in opt['datasets'].items():
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = lq_folder
        if dataset.get('dataroot_gt') is not None:
            # only set if GT folder exists
            if osp.exists(gt_folder):
                dataset['dataroot_gt'] = gt_folder

    # ensure result path points to output_path
    opt['path']['results_root'] = output_path
    opt['is_train'] = False

    # ensure pretrained path points to provided model_dir if given
    if model_dir is not None:
        opt['path']['pretrain_network_g'] = model_dir

    # build dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt.get('num_gpu', 1), dist=opt.get('dist', False), sampler=None, seed=opt.get('manual_seed', 0))
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # run inference and save to output_path (flat folder)
    os.makedirs(output_path, exist_ok=True)

    from basicsr.utils import tensor2img, imwrite

    for test_loader in test_loaders:
        for idx, val_data in enumerate(test_loader):
            model.feed_data(val_data)
            t0 = time.time()
            model.test()
            t1 = time.time()

            visuals = model.get_current_visuals()
            sr_img = visuals['result']
            out_img = tensor2img([sr_img])

            lq_path = val_data['lq_path'][0]
            basename = os.path.basename(lq_path)
            save_path = osp.join(output_path, basename)
            if not osp.splitext(save_path)[1].lower() in ['.png', '.tif', '.tiff']:
                save_path = osp.splitext(save_path)[0] + '.png'

            imwrite(out_img, save_path)

    logger.info(f'Inference finished. Results saved to {output_path}')
