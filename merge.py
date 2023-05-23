#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from options import merge_setting
from lib.predict.segmentation_export import save_segmentation_nifti_from_softmax
from lib.predict.evaluator import evaluate_folder

# from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing

from batchgenerators.utilities.file_and_folder_operations import *

from glob import glob
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
import SimpleITK as sitk


def main():
    args = merge_setting().parse_args()

    input_folder_list = [f'{args.default_output_folder}/{args.task}/{i}' for i in args.input_list]
    output_folder_name = os.path.join(args.default_output_folder, args.task, args.output_name) #os.path.join(args.output_folder, args.task)  

    if args.method == 'merge':
        merge(input_folder_list, output_folder_name, args.threads, override=True, postprocessing_file=args.postprocessing_file, store_npz=args.npz)
    else:
        union(input_folder_list, output_folder_name)

    if args.gt_name != None:
        gt_folder_name = os.path.join(args.default_dataset_folder, args.task, args.gt_name) 
        evaluate_folder(gt_folder_name, output_folder_name, args.labels)

def union(input_folder_list, output_folder_name):
    input_files = []
    for input_folder in input_folder_list:
        input_files.append(sorted(glob(f'{input_folder}/*.nii.gz')))
    for idx in range(len(input_files[0])):
        id = input_files[0][idx].split('/')[-1]
        
        prediction = []
        for idx2 in range(len(input_folder_list)):
            prediction.append(sitk.ReadImage(input_files[idx2][idx]))
        origin, spacing, direction = prediction[0].GetOrigin(), prediction[0].GetSpacing(), prediction[0].GetDirection()
        for idx2 in range(len(input_folder_list)-1):
            if idx2 == 0:
                union_ = sitk.GetArrayFromImage(prediction[idx2])
            union_ = np.logical_or(union_, sitk.GetArrayFromImage(prediction[idx2+1])) #union
        union_ = np.logical_or(sitk.GetArrayFromImage(prediction[0]), sitk.GetArrayFromImage(prediction[1])) #union
        union = np.squeeze(union_).astype(int)
        union_image = sitk.GetImageFromArray(union)
        union_image.SetOrigin(origin), union_image.SetSpacing(spacing), union_image.SetDirection(direction)
        sitk.WriteImage(union_image, f'{output_folder_name}/{id}')

def merge_files(files, properties_files, out_file, override, store_npz):
    if override or not isfile(out_file):
        softmax = [np.load(f)['softmax'][None] for f in files]
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0)
        props = [load_pickle(f) for f in properties_files]

        reg_class_orders = [p['regions_class_order'] if 'regions_class_order' in p.keys() else None
                            for p in props]

        if not all([i is None for i in reg_class_orders]):
            # if reg_class_orders are not None then they must be the same in all pkls
            tmp = reg_class_orders[0]
            for r in reg_class_orders[1:]:
                assert tmp == r, 'If merging files with regions_class_order, the regions_class_orders of all ' \
                                 'files must be the same. regions_class_order: %s, \n files: %s' % \
                                 (str(reg_class_orders), str(files))
            regions_class_order = tmp
        else:
            regions_class_order = None

        # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
        # don't matter here)
        save_segmentation_nifti_from_softmax(softmax, out_file, props[0], 3, regions_class_order, None, None,
                                             force_separate_z=None)
        if store_npz:
            np.savez_compressed(out_file[:-7] + ".npz", softmax=softmax)
            save_pickle(props, out_file[:-7] + ".pkl")


def merge(folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):
    maybe_mkdir_p(output_folder)

    if postprocessing_file is not None:
        output_folder_orig = deepcopy(output_folder)
        output_folder = join(output_folder, 'not_postprocessed')
        maybe_mkdir_p(output_folder)
    else:
        output_folder_orig = None

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    for f in folders:
        assert all([isfile(join(f, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                        "all folders"
        assert all([isfile(join(f, i + ".pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append([join(f, p + ".npz") for f in folders])
        property_files.append([join(f, p + ".pkl") for f in folders])
        out_files.append(join(output_folder, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(merge_files, zip(files, property_files, out_files, [override] * len(out_files), [store_npz] * len(out_files)))
    p.close()
    p.join()

    # if postprocessing_file is not None:
    #     for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
    #     print('Postprocessing...')
    #     apply_postprocessing_to_folder(output_folder, output_folder_orig,
    #                                    for_which_classes, min_valid_obj_size, threads)
    #     shutil.copy(postprocessing_file, output_folder_orig)


if __name__ == "__main__":
    main()