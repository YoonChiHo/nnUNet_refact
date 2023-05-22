import argparse
import os

default_num_threads = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])
default_data_identifier = 'nnUNetData_plans_v2.1'
RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3  # determines what threshold to use for resampling the low resolution axis

def base_setting(parser):
    parser.add_argument("-t", "--task", default= 'Task500_ISLES_ad')#required=True)
    parser.add_argument('--network', help="2d, 3d_fullres. ",default="3d_fullres", required=False)

    parser.add_argument("--dataset_dir", default= '/data/1_nnunet_refactoring/nnUNet_refact/Dataset')#required=True)
    parser.add_argument("--preprocess_dir", default= '/data/1_nnunet_refactoring/nnUNet_refact/Preprocessed')#required=True)
    parser.add_argument("--checkpoints", default= '/data/1_nnunet_refactoring/nnUNet_refact/Checkpoints')#required=True)
    parser.add_argument('-tr', '--network_trainer',required=False,default='nnUNetTrainerV2')
    parser.add_argument("--plans_identifier", default='nnUNetPlansv2.1') 

def preprocess_setting():
    parser = argparse.ArgumentParser()
    base_setting(parser)

    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")
    return parser

def train_setting():
    parser = argparse.ArgumentParser()
    base_setting(parser)

    parser.add_argument("--do_preprocess", default = True)
    parser.add_argument("--max_epoch", default= 1)#required=True)
    parser.add_argument("--i_lr", default= 1e-2)#required=True)
    parser.add_argument("-f", "--fold", default= '0')#required=True)
    parser.add_argument("-d","--deterministic", default=True) #,action='store_true')#required=True)
    parser.add_argument("-c", "--continue_training", default=False, required=False)
    parser.add_argument("-vd", "--val_data", default=[], required=False)
    parser.add_argument("-g", "--gpus", default="0") #"0,1,2,3" "0"
    parser.add_argument("-p", "--pretrained_weights", default=None) #"nnUNet_trained_models/nnUNet/3d_fullres/Task505_BRATS/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model",
    #fixed parmeters
    parser.add_argument("--decompress_data", default=True) 
    parser.add_argument("--batch_dice", default=False) 
    return parser

def test_setting():
    parser = argparse.ArgumentParser()
    base_setting(parser)

    parser.add_argument("-i", '--input_folder', default = "/data/1_nnunet_refactoring/nnUNet_refact/Task500_ISLES_ad/imagesTs")#, required=True)
    parser.add_argument('-o', "--output_folder", default = "/data/1_nnunet_refactoring/nnUNet_refact/Results",
                        required=False, help="folder for saving predictions")
    parser.add_argument('-g', "--gt_folder", default = "/data/1_nnunet_refactoring/nnUNet_refact/Task500_ISLES_ad/labelsTs",
                     required=False, help="folder for saving predictions")
    parser.add_argument('-l', "--labels", default = [0,1])
    parser.add_argument('-m', "--metrics", default = ["Dice"])
    parser.add_argument('-f', '--fold', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % "nnUNetTrainerV2CascadeFullRes", required=False,
                        default="nnUNetTrainerV2CascadeFullRes")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                        help="use this if you want to ensemble these predictions with those of other models. Softmax "
                             "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                             "merged between output_folders with nnUNet_ensemble_predictions")
    # parser.add_argument('-l', '--lowres_segmentations', required=False, default='None',
    #                     help="if model is the highres stage of the cascade then you can use this folder to provide "
    #                          "predictions from the low resolution 3D U-Net. If this is left at default, the "
    #                          "predictions will be generated automatically (provided that the 3D low resolution U-Net "
    #                          "network weights are present")
    parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                                                               "the folder over several GPUs. If you "
                                                                               "want to use n GPUs to predict this "
                                                                               "folder you need to run this command "
                                                                               "n times with --part_id=0, ... n-1 and "
                                                                               "--num_parts=n (each with a different "
                                                                               "GPU (for example via "
                                                                               "CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_parts", type=int, required=False, default=1, help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you want to use n GPUs to predict this "
                             "folder you need to run this command n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different GPU (via CUDA_VISIBLE_DEVICES=X)")
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
    "Determines many background processes will be used for data preprocessing. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 6")
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
    "Determines many background processes will be used for segmentation export. Reduce this if you "
    "run into out of memory (RAM) problems. Default: 2")
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="set this flag to disable test time data augmentation via mirroring. Speeds up inference "
                             "by roughly factor 4 (2D) or 8 (3D)")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true",
                        help="Set this flag if the target folder contains predictions that you would like to overwrite")
    parser.add_argument("--mode", type=str, default="normal", required=False, help="Hands off!")
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. " "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    parser.add_argument('-chk', help='checkpoint name, default: model_final_checkpoint', required=False, default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')
    
    return parser