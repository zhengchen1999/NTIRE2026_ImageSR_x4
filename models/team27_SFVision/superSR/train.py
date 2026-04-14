# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

from . import archs  # noqa: F401
from . import data   # noqa: F401
from . import models # noqa: F401

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
