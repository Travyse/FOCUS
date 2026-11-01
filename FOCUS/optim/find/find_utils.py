# Inlined from FIND (https://github.com/OllieBoyne/FIND, MIT licensed - see
# FIND_LICENSE.md). Original read `src/cfg.yaml` relative to the FIND repo
# layout; here it reads `cfg.yaml` bundled next to this module.

import os

import torch
import yaml


def up1(p):
    return os.path.split(p)[0]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cfg.yaml')) as infile:
    cfg = yaml.load(infile, Loader=yaml.FullLoader)