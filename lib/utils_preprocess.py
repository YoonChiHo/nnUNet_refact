import numpy as np
from copy import deepcopy
import os
import shutil
import json

from lib.cropping import ImageCropper

default_num_threads = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis
# separately (with NN)
def get_pool_and_conv_props_poolLateV2(patch_size, min_feature_map_size, max_numpool, spacing):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    initial_spacing = deepcopy(spacing)
    reach = max(initial_spacing)
    dim = len(patch_size)

    num_pool_per_axis = get_network_numpool(patch_size, max_numpool, min_feature_map_size)

    net_num_pool_op_kernel_sizes = []
    net_conv_kernel_sizes = []
    net_numpool = max(num_pool_per_axis)

    current_spacing = spacing
    for p in range(net_numpool):
        reached = [current_spacing[i] / reach > 0.5 for i in range(dim)]
        pool = [2 if num_pool_per_axis[i] + p >= net_numpool else 1 for i in range(dim)]
        if all(reached):
            conv = [3] * dim
        else:
            conv = [3 if not reached[i] else 1 for i in range(dim)]
        net_num_pool_op_kernel_sizes.append(pool)
        net_conv_kernel_sizes.append(conv)
        current_spacing = [i * j for i, j in zip(current_spacing, pool)]

    net_conv_kernel_sizes.append([3] * dim)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    return num_pool_per_axis, net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, patch_size, must_be_divisible_by


def get_pool_and_conv_props(spacing, patch_size, min_feature_map_size, max_numpool):
    """

    :param spacing:
    :param patch_size:
    :param min_feature_map_size: min edge length of feature maps in bottleneck
    :return:
    """
    dim = len(spacing)

    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = []
    conv_kernel_sizes = []

    num_pool_per_axis = [0] * dim

    while True:
        # This is a problem because sometimes we have spacing 20, 50, 50 and we want to still keep pooling.
        # Here we would stop however. This is not what we want! Fixed in get_pool_and_conv_propsv2
        min_spacing = min(current_spacing)
        valid_axes_for_pool = [i for i in range(dim) if current_spacing[i] / min_spacing < 2]
        axes = []
        for a in range(dim):
            my_spacing = current_spacing[a]
            partners = [i for i in range(dim) if current_spacing[i] / my_spacing < 2 and my_spacing / current_spacing[i] < 2]
            if len(partners) > len(axes):
                axes = partners
        conv_kernel_size = [3 if i in axes else 1 for i in range(dim)]

        # exclude axes that we cannot pool further because of min_feature_map_size constraint
        #before = len(valid_axes_for_pool)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_size[i] >= 2*min_feature_map_size]
        #after = len(valid_axes_for_pool)
        #if after == 1 and before > 1:
        #    break

        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 0:
            break

        #print(current_spacing, current_size)

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]

        pool_kernel_sizes = [0] * dim
        for v in valid_axes_for_pool:
            pool_kernel_sizes[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = np.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernel_sizes[nv] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(conv_kernel_size)
        #print(conv_kernel_sizes)

    must_be_divisible_by = get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = pad_shape(patch_size, must_be_divisible_by)

    # we need to add one more conv_kernel_size for the bottleneck. We always use 3x3(x3) conv here
    conv_kernel_sizes.append([3]*dim)
    return num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, must_be_divisible_by

def get_shape_must_be_divisible_by(net_numpool_per_axis):
    return 2 ** np.array(net_numpool_per_axis)

def pad_shape(shape, must_be_divisible_by):
    """
    pads shape so that it is divisibly by must_be_divisible_by
    :param shape:
    :param must_be_divisible_by:
    :return:
    """
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    else:
        assert len(must_be_divisible_by) == len(shape)

    new_shp = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]

    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    new_shp = np.array(new_shp).astype(int)
    return new_shp


def get_network_numpool(patch_size, maxpool_cap=999, min_feature_map_size=4):
    network_numpool_per_axis = np.floor([np.log(i / min_feature_map_size) / np.log(2) for i in patch_size]).astype(int)
    network_numpool_per_axis = [min(i, maxpool_cap) for i in network_numpool_per_axis]
    return network_numpool_per_axis

def crop(task_string, nnUNet_cropped_data, nnUNet_raw_data, override=False, num_threads=default_num_threads):
    cropped_out_dir = os.path.join(nnUNet_cropped_data, task_string)
    #maybe_mkdir_p(cropped_out_dir)
    os.makedirs(cropped_out_dir, exist_ok=True)
    if override and os.path.isdir(cropped_out_dir):
        shutil.rmtree(cropped_out_dir)
        #maybe_mkdir_p(cropped_out_dir)
        os.makedirs(cropped_out_dir, exist_ok=True)

    splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, task_string)
    lists, _ = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    imgcrop = ImageCropper(num_threads, cropped_out_dir)
    imgcrop.run_cropping(lists, overwrite_existing=override)
    shutil.copy(os.path.join(nnUNet_raw_data, task_string, "dataset.json"), cropped_out_dir)

    
def create_lists_from_splitted_dataset(base_folder_splitted):
    lists = []

    json_file = os.path.join(base_folder_splitted, "dataset.json")
    with open(json_file) as jsn:
        d = json.load(jsn)
        training_files = d['training']
    num_modalities = len(d['modality'].keys())
    for tr in training_files:
        cur_pat = []
        for mod in range(num_modalities):
            cur_pat.append(os.path.join(base_folder_splitted, "imagesTr", tr['image'].split("/")[-1][:-7] +
                                "_%04.0d.nii.gz" % mod))
        cur_pat.append(os.path.join(base_folder_splitted, "labelsTr", tr['label'].split("/")[-1]))
        lists.append(cur_pat)
    return lists, {int(i): d['modality'][str(i)] for i in d['modality'].keys()}