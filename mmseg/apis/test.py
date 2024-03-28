import os.path as osp
import pickle
import shutil
import tempfile
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import wandb

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmseg.utils import split_images


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    wandb_run = None,
                    opacity=0.5):
    """Test with single GPU.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    gt_bc_maps = dataset.get_gt_bc_maps()
    gt_sem_maps = dataset.get_gt_sem_maps()
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            if i % 60 == 0:
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]

                imgs = []

                binary_image = Image.fromarray((result['bc'] * 255).astype(np.uint8), mode='L')
                num_classes = len(np.unique(gt_sem_maps[i]))
                colors = plt.cm.get_cmap('gray', num_classes)
                colored_mask = colors(result['sem'])
                sem_image = Image.fromarray((colored_mask[:, :, 0] * 255).astype(np.uint8), mode='L')
                input_t1_path = dataset.img_infos[i]['filename_pre']
                input_t1_array = mmcv.imread(
                input_t1_path, flag='unchanged', backend='tifffile')
                input_t2_path = dataset.img_infos[i]['filename']
                input_t2_array = mmcv.imread(
                input_t2_path, flag='unchanged', backend='tifffile')

                input_t1_img = Image.fromarray((input_t1_array).astype(np.uint8))
                input_t2_img = Image.fromarray((input_t2_array).astype(np.uint8))

                binary_gt = Image.fromarray((gt_bc_maps[i] * 255).astype(np.uint8), mode='L')
                colored_mask_gt = colors(gt_sem_maps[i])
                sem_gt = Image.fromarray((colored_mask_gt[:, :, 0] * 255).astype(np.uint8), mode='L')

                cols = 2
                rows = 3
                imgs.extend([input_t1_img, input_t2_img, binary_gt, sem_gt, binary_image, sem_image])
                w, h = imgs[0].size
                caption_height = 180
                grid = Image.new('RGB', size=(cols*w, rows*(h + caption_height)))
                draw = ImageDraw.Draw(grid)
                font = ImageFont.truetype("arial.ttf", 100)
                caption = ["Image T1", "Image T2", "BC GT", "Sem Seg GT", "BC Pred", "Sem Seg Pred"]
    
                for i, img in enumerate(imgs):
                    grid.paste(img, box=(i%cols*w, i//cols*(h+caption_height)))
                    draw.text((i%cols*w, (i // cols + 1) * (h + caption_height) - caption_height + 10), caption[i], font=font, fill=(255, 255, 255))

                caption_text = input_t1_path.split('2006', 1)

                wandb_run.log({
                    "input_gt_pred": wandb.Image(grid, caption=f'{caption_text[1]}'),
                })
            
            # imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            # assert len(imgs) == len(img_metas)

            # for img, img_meta in zip(imgs, img_metas):
            #     h, w, _ = img_meta['img_shape']
            #     img_show = img[:h, :w, :]

            #     ori_h, ori_w = img_meta['ori_shape'][:-1]
            #     img_show = mmcv.imresize(img_show, (ori_w, ori_h))

            #     if out_dir:
            #         out_file = osp.join(out_dir, img_meta['ori_filename'])
            #     else:
            #         out_file = None

            #     model.module.show_result(
            #         img_show,
            #         result,
            #         palette=dataset.PALETTE,
            #         show=show,
            #         out_file=out_file,
            #         opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data_loader.batch_size # len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results
    

#def single_gpu_test(model,
#                    data_loader,
#                    show=False,
#                    out_dir=None,
#                    efficient_test=False,
#                    opacity=0.5):
#    """Test with single GPU.
#
#    Args:
#        model (nn.Module): Model to be tested.
#        data_loader (utils.data.Dataloader): Pytorch data loader.
#        show (bool): Whether show results during inference. Default: False.
#        out_dir (str, optional): If specified, the results will be dumped into
#            the directory to save output results.
#        efficient_test (bool): Whether save the results as local numpy files to
#            save CPU memory during evaluation. Default: False.
#        opacity(float): Opacity of painted segmentation map.
#            Default 0.5.
#            Must be in (0, 1] range.
#    Returns:
#        list: The prediction results.
#    """
#
#    model.eval()
#    results = []
#    dataset = data_loader.dataset
#    prog_bar = mmcv.ProgressBar(len(dataset))
#    for i, data in enumerate(data_loader):
#        with torch.no_grad():
#            result = model(return_loss=False, **data)
#        batch_size = len(result)
#        if show or out_dir:
#            img_tensor, _ = split_images(data['img'][0])
#            img_metas = data['img_metas'][0].data[0]
#            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#            assert len(imgs) == len(img_metas)
#
#            for index, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                h, w, _ = img_meta['img_shape']
#                img_show = img[:h, :w, :]
#
#                ori_h, ori_w = img_meta['ori_shape'][:-1]
#                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
#
#                if out_dir:
#                    out_file = osp.join(out_dir, img_meta['ori_filename'])
#                else:
#                    out_file = None
#                model.module.show_result(
#                    img_show,
#                    result,
#                    index=index,
#                    palette=dataset.PALETTE,
#                    show=show,
#                    out_file=out_file,
#                    opacity=opacity)
#
#        for _ in range(batch_size):
#            prog_bar.update()
#    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
