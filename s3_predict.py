from options import test_setting, format
from lib.predict.predict_cases import predict_cases, check_input_folder_and_return_caseIDs
from lib.predict.evaluator import evaluate_folder

from batchgenerators.utilities.file_and_folder_operations import subfiles, save_json, load_pickle

from time import time
import shutil
import os


def main():
    #Main Parameter Settings
    args = test_setting().parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # "0, 1, 2, 3"

    # if not args.task_name.startswith("Task"):
    #     task_id = int(args.task_name)
    #     task_name = convert_id_to_task_name(task_id)
    # else:
    #     task_name = args.task_name

    assert args.network in ["2d",  "3d_fullres"], "-m must be 2d, 3d_fullres"

    # Set Directory
    model_folder_name = os.path.join(args.default_checkpoints_folder, args.network, args.task) 
    input_folder_name = os.path.join(args.default_dataset_folder, args.task, args.input_name) 
    #model_folder_name = os.path.join(checkpoint_folder_name, args.network, args.task)
    print("using model stored in ", model_folder_name)
    assert os.path.isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    output_folder_name = os.path.join(args.default_output_folder, args.task, args.output_name) #os.path.join(args.output_folder, args.task)  
    os.makedirs(output_folder_name, exist_ok=True)

    # if lowres_segmentations == "None":
    lowres_segmentations = None

    if isinstance(args.fold, list):
        if args.fold[0] == 'all' and len(args.fold) == 1:
            fold=args.fold
        else:
            fold = [int(i) for i in args.fold]
    elif args.fold == "None":
        fold = None
    else:
        fold = args.fold

    assert args.all_in_gpu in ['None', 'False', 'True']
    if args.all_in_gpu == "None":
        all_in_gpu = None
    elif args.all_in_gpu == "True":
        all_in_gpu = True
    elif args.all_in_gpu == "False":
        all_in_gpu = False

    # we need to catch the case where model is 3d cascade fullres and the low resolution folder has not been set.
    # In that case we need to try and predict with 3d low res first
    # if model == "3d_cascade_fullres":# and lowres_segmentations is None:
    #     print("lowres_segmentations is None. Attempting to predict 3d_lowres first...")
    #     assert part_id == 0 and num_parts == 1, "if you don't specify a --lowres_segmentations folder for the " \
    #                                             "inference of the cascade, custom values for part_id and num_parts " \
    #                                             "are not supported. If you wish to have multiple parts, please " \
    #                                             "run the 3d_lowres inference first (separately)"
    #     model_folder_name = os.path.join(network_training_output_dir, "3d_lowres", task_name, trainer_class_name + "__" +
    #                               args.plans_identifier)
    #     assert os.path.isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name
    #     lowres_output_folder = os.path.join(output_folder, "3d_lowres_predictions")
    #     predict_from_folder(model_folder_name, input_folder, lowres_output_folder, folds, False,
    #                         num_threads_preprocessing, num_threads_nifti_save, None, part_id, num_parts, not disable_tta,
    #                         overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
    #                         mixed_precision=not args.disable_mixed_precision,
    #                         step_size=step_size)
    #     lowres_segmentations = lowres_output_folder
    #     torch.cuda.empty_cache()
    #     print("3d_lowres done")

    # if model == "3d_cascade_fullres":
    #     trainer = cascade_trainer_class_name
    # else:

    st = time()
    # predict_from_folder(model_folder_name, args.input_folder, args.output_folder, fold, args.save_npz, args.num_threads_preprocessing,
    #                     args.num_threads_nifti_save, lowres_segmentations, args.part_id, args.num_parts, not args.disable_tta,
    #                     overwrite_existing=args.overwrite_existing, mode=args.mode, overwrite_all_in_gpu=all_in_gpu,
    #                     mixed_precision=not args.disable_mixed_precision,
    #                     step_size=args.step_size, checkpoint_name=args.chk)


    shutil.copy(os.path.join(model_folder_name, 'plans.pkl'), output_folder_name)

    assert os.path.isfile(os.path.join(model_folder_name, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(os.path.join(model_folder_name, "plans.pkl"))['num_modalities']

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(input_folder_name, expected_num_modalities)

    output_files = [os.path.join(output_folder_name, i + f".{format}") for i in case_ids]
    all_files = subfiles(input_folder_name, suffix=f".{format}", join=False, sort=True)
    list_of_lists = [[os.path.join(input_folder_name, i) for i in all_files if i[:len(j)].startswith(j) and
                        len(i) == (len(j) + 6 + len(format))] for j in case_ids]

    # if lowres_segmentations is not None:
    #     assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
    #     lowres_segmentations = [os.path.join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
    #     assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " \
    #                                                            "(I was searching for case_id.nii.gz in that folder)"
    #     lowres_segmentations = lowres_segmentations[part_id::num_parts]
    # else:
    lowres_segmentations = None

    #if mode == "normal": default
    #if args.overwrite_all_in_gpu is None:
    all_in_gpu = False
    # else:
    #     all_in_gpu = args.overwrite_all_in_gpu

    predict_cases(model_folder_name, list_of_lists[args.part_id::args.num_parts], output_files[args.part_id::args.num_parts], fold,
                        args.save_npz, args.num_threads_preprocessing, args.num_threads_nifti_save, lowres_segmentations, not args.disable_tta,
                        mixed_precision=not args.disable_mixed_precision, overwrite_existing=args.overwrite_existing,
                        all_in_gpu=all_in_gpu,
                        step_size=args.step_size, checkpoint_name=args.chk)

    end = time()
    save_json(end - st, os.path.join(output_folder_name, 'prediction_time.txt'))

    if args.gt_name != None:
        gt_folder_name = os.path.join(args.default_dataset_folder, args.task, args.gt_name) 
        evaluate_folder(gt_folder_name, output_folder_name, args.labels)

if __name__ == "__main__":
    main()