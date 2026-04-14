import os
import glob
import cv2
import torch

from .hat_ffl_model import Model


def model_func(model_dir, input_path, output_path, device):
    model = Model(model_dir, device)

    paths = sorted(glob.glob(os.path.join(input_path, '*')))

    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out = model.process(img)

        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        name = os.path.basename(p)
        cv2.imwrite(os.path.join(output_path, name), out)
