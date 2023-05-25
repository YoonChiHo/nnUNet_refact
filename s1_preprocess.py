
from options import preprocess_setting, format
from lib.preprocess.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21 as planner_3d
from lib.preprocess.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21 as planner_2d
from lib.preprocess.DatasetAnalyzer import DatasetAnalyzer
from lib.preprocess.cropping import crop

from batchgenerators.utilities.file_and_folder_operations import load_json, subfiles, join, save_json

from glob import glob
import shutil 
from typing import Tuple, List
import os
import numpy as np

def main():
    #Main Parameter Settings
    args = preprocess_setting().parse_args()

    #if args.planner3d == "None": planner_name3d = None
    #else: planner_name3d = args.planner3d
        
    #if args.planner3d == "None": planner_name2d = None
    #else: planner_name2d = args.planner2d

    # if args.overwrite_plans is not None:
    #     if planner_name2d is not None:
    #         print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
    #                 "skip 2d planning and preprocessing.")
    #     assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
    #                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    #tasks = [args.task]
    # Generater _0000 format style Labels
    labels_name = glob(join(args.default_dataset_folder, args.task, 'imagesTr','*'))
    if len(labels_name) != 0:
        if labels_name[0][-5-len(format):-1-len(format)] != '0000':
            for g in labels_name:
                os.rename(g, f'{g[:-1-len(format)]}_0000.{format}')
        else:
            print(f'no need to change format')
    labels_name = glob(join(args.default_dataset_folder, args.task, 'imagesTs','*'))
    if len(labels_name) != 0:
        if labels_name[0][-5-len(format):-1-len(format)] != '0000':
            for g in labels_name:
                os.rename(g, f'{g[:-1-len(format)]}_0000.{format}')
            else:
                print(f'no need to change format')

    # # Generater mask to data format style 
    # labels_name = glob(join(args.default_dataset_folder, args.task, 'labelsTr','*'))
    # if len(labels_name) != 0:
    #     if labels_name[0].split('/')[-1][:4] != 'DATA':
    #         for g in labels_name:
    #             gsp = g.split('/')
    #             os.rename(g, f'{g[:-14]}/DATA{gsp[-1][4:]}')
    #     else:
    #         print(f'no need to change format')
    # labels_name = glob(join(args.default_dataset_folder, args.task, 'labelsTs','*'))
    # if len(labels_name) != 0:
    #     if labels_name[0].split('/')[-1][:4] != 'DATA':
    #         for g in labels_name:
    #             gsp = g.split('/')
    #             os.rename(g, f'{g[:-14]}/DATA{gsp[-1][4:]}')
    #         else:
    #             print(f'no need to change format')

    # Generate dataset.json
    generate_dataset_json(join(args.default_dataset_folder,args.task,'dataset.json'), join(args.default_dataset_folder, args.task, 'imagesTr'), join(args.default_dataset_folder, args.task, 'imagesTs'),('tdsc',), #('adc', 'dwi', 'flair'),
                        labels={0: 'background', 1: 'label'}, dataset_name=args.task, license='hands off!')


    print("\n\n\n", args.task)
        #parser.add_argument("--cropped_out_dir", default= '/data/1_nnunet_refactoring/nnUNet_raw_data_base/nnUNet_cropped_data')#required=True)

    cropped_out_dir = os.path.join(args.default_preprocessed_folder, args.task, 'cropped_data')
    crop(args.task, cropped_out_dir, args.default_dataset_folder , False, args.tf)
    
    
    
    #cropped_out_dir = os.path.join(cropped_out_dir, args.task)
    preprocessing_output_dir_this_task = os.path.join(args.default_preprocessed_folder, args.task)
    #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
    #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

    # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
    dataset_json = load_json(os.path.join(cropped_out_dir, 'dataset.json'))
    modalities = list(dataset_json["modality"].values())
    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
    dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=args.tf)  # this class creates the fingerprint
    _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


    #maybe_mkdir_p(preprocessing_output_dir_this_task)
    os.makedirs(preprocessing_output_dir_this_task, exist_ok=True)

    shutil.copy(os.path.join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
    shutil.copy(os.path.join(args.default_dataset_folder, args.task, "dataset.json"), preprocessing_output_dir_this_task)

    threads = (args.tl, args.tf)

    print("number of threads: ", threads, "\n")

    if planner_3d is not None:
        if args.overwrite_plans is not None:
            assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
                                        args.overwrite_plans_identifier)
        else:
            exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.patchsize_3d)
        exp_planner.plan_experiment()
        if not args.no_pp:  # double negative, yooo
            exp_planner.run_preprocessing(threads)
    if planner_2d is not None:
        exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task, args.patchsize_2d)
        exp_planner.plan_experiment()
        if not args.no_pp:  # double negative, yooo
            exp_planner.run_preprocessing(threads)



def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-6-len(format)] for i in subfiles(folder, suffix=f'.{format}', join=False)]) #-12 FOR NIFTI

    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.%s" % (i,format), "label": "./labelsTr/%s.%s" % (i,format)} for i
        in
        train_identifiers]
    #json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]
    json_dict['test'] = ["./imagesTs/%s.%s" % (i,format) for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)

if __name__ == "__main__":
    main()