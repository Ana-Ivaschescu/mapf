import os
import sys
sys.path.append(os.getcwd())

import argparse
import mmcv
import torch
import rasterio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.apis import fast_single_gpu_test_sp, fast_single_gpu_test_mp
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--fast-test',
        type=str,
        default='false',
        help='Parallel: sp or mp')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=16,
        help='Custom option for batch size.'
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def save_mask_as_geotiff(mask, save_path, num_channels):
    # Assuming a rasterio-compatible profile, you may need to customize based on your data
    profile = {
        'count': num_channels,  # 1 channel for the mask
        'dtype': 'uint8',
        'driver': 'GTiff',
        'width': 2000,
        'height': 2000,
        'crs': 'EPSG:2154',  # Update with the appropriate CRS
        'transform': rasterio.Affine(1.0, 0, 0, 0, -1.0, mask.shape[0])
    }

    mask_reshaped = mask.reshape(profile['height'], profile['width'])
    # Write the mask using Rasterio
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(mask_reshaped, 1)


def save_predicted_masks(results, save_dir, dataset, opacity=0.5):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(results)):
        filename = f'image_{i}'  # You can modify this to create a meaningful filename

        # Assuming 'bc' and 'sem' keys exist in the results
        pred_bc = results[i]['bc']
        print(pred_bc.size)
        pred_sem = results[i]['sem']
        print(pred_sem.size)

        # Save predicted BC mask
        save_path_bc = os.path.join(save_dir, f'{filename}_bc_mask.tif')
        save_mask_as_geotiff(pred_bc, save_path_bc, num_channels=1)

        # Save predicted semantic mask
        save_path_sem = os.path.join(save_dir, f'{filename}_sem_mask.tif')
        save_mask_as_geotiff(pred_sem, save_path_sem, num_channels=5)


def main(args=None):
    if args is None:
        args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # if args.aug_test:
    #     # hard code index
    #     # cfg.data.test.pipeline[1].img_ratios = [0.75, 1.0, 1.75]
    #     samples_per_gpu = 16
    #     # cfg.data.test.pipeline[1].img_ratios = [
    #     #    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    #     # ]
    #     cfg.data.test.pipeline[1].flip = True
    # else:
    #     samples_per_gpu = 16
    samples_per_gpu = args.samples_per_gpu

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', False)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        # print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        if args.fast_test == 'sp':
            outputs = fast_single_gpu_test_sp(model, data_loader, args.show_dir)
        elif args.fast_test == 'mp':
            outputs = fast_single_gpu_test_mp(model, data_loader, args.show_dir)
        else:
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    #print(np.unique(outputs[0]['sem']))
    #print(outputs[0]['sem'])
    #print(np.sum(outputs[0]['bc']))
    gt_bc_maps = dataset.get_gt_bc_maps()
    gt_sem_maps = dataset.get_gt_sem_maps()
    # print(gt_sem_maps[0])
    num_classes = len(np.unique(gt_sem_maps[0]))
    # print(num_classes)
    # print(np.unique(gt_sem_maps[0]))
    # print(np.sum(gt_bc_maps[0]))
    if cfg.plot_test:
        for i, mask in enumerate(outputs):
            num_classes = len(np.unique(gt_sem_maps[i]))
            colors = plt.cm.get_cmap('gray', num_classes)
            colored_mask = colors(mask['sem'])
            sem_image = Image.fromarray((colored_mask[:, :, 0] * 255).astype(np.uint8), mode='L')
            sem_image.save(f"sem_mask_{i}.png")
            binary_image = Image.fromarray((mask['bc'] * 255).astype(np.uint8), mode='L')
            binary_image.save(f"bc_mask_{i}.png")
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            dataset.evaluate(outputs, args.eval, **kwargs)
        
        #show_dir = "./output_2/"
        #save_predicted_masks(outputs, show_dir, dataset, args.opacity)


if __name__ == '__main__':
    main()
