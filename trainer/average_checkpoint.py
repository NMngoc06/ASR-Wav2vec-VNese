# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import glob

import yaml
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    checkpoints = []
    val_scores = []

    path_list = glob.glob('{}/model_[0-9]*.pth'.format(args.src_path))
    # path_list = sorted(path_list)
    path_list = sorted(path_list, key=os.path.getmtime)
    path_list = path_list[-args.num:]

    state_dict = {
        "epoch": -1,
        "best_score": -1,
        "completed_steps": -1
    }

    avg = None
    num = args.num
    assert num == len(path_list)
    for idx, path in enumerate(path_list):
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states["model"]
        else:
            for k in avg.keys():
                avg[k] += states["model"][k]
        
        if idx == len(path_list) - 1:
            state_dict['epoch'] = states['epoch']
            state_dict['best_score'] = states['best_score']
            state_dict['completed_steps'] = states['completed_steps']
            state_dict['scheduler'] = states['scheduler']

    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)
    
    state_dict['model'] = avg
    dst_model = os.path.join(args.src_path, f"avg_{num}.pth")
    print('Saving to {}'.format(dst_model))
    torch.save(state_dict, dst_model)


if __name__ == '__main__':
    main()
