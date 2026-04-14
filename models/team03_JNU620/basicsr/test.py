import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import time
import zipfile
import os


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
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

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        # run per-image inference and save outputs to a flat 'res' folder
        res_dir = os.path.join(os.getcwd(), 'res')
        os.makedirs(res_dir, exist_ok=True)

        times = []
        for idx, val_data in enumerate(test_loader):
            model.feed_data(val_data)
            t0 = time.time()
            model.test()
            t1 = time.time()
            times.append(t1 - t0)

            visuals = model.get_current_visuals()
            sr_img = visuals['result']  # tensor
            from basicsr.utils import tensor2img, imwrite
            out_img = tensor2img([sr_img])

            # keep original input basename (with extension)
            lq_path = val_data['lq_path'][0]
            basename = os.path.basename(lq_path)
            save_path = os.path.join(res_dir, basename)
            # ensure PNG (lossless) extension
            if not os.path.splitext(save_path)[1].lower() in ['.png', '.tif', '.tiff']:
                save_path = os.path.splitext(save_path)[0] + '.png'

            imwrite(out_img, save_path)

        # compute average runtime per image
        avg_time = sum(times) / len(times) if len(times) > 0 else 0.0

        # determine device flag: CPU[1]/GPU[0]
        device_flag = 0 if (torch.cuda.is_available() and opt.get('num_gpu', 0) != 0) else 1
        # extra data flag from options (default 0)
        extra_flag = 1 if opt.get('extra_data', False) else 0

        # write readme.txt inside res_dir
        readme_path = os.path.join(res_dir, 'readme.txt')
        with open(readme_path, 'w') as f:
            f.write(f"runtime per image [s] : {avg_time:.4f}\n")
            f.write(f"CPU[1] / GPU[0] : {device_flag}\n")
            f.write(f"Extra Data [1] / No Extra Data [0] : {extra_flag}\n")
            f.write("Other description : \n")

        # create zip archive with flat structure (no folders inside)
        zip_path = os.path.join(os.getcwd(), 'res.zip')
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as z:
            for fname in sorted(os.listdir(res_dir)):
                fpath = os.path.join(res_dir, fname)
                if os.path.isfile(fpath):
                    z.write(fpath, arcname=fname)

        logger.info(f'Results saved to {res_dir} and zipped to {zip_path}')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
