import argparse
import os


def initial_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default= '/data/1_nnunet_refactoring/nnUNet_preprocessed')#required=True)
    parser.add_argument("--checkpoints", default= '/data/1_nnunet_refactoring/nnUNet_refact/nnUNet_trained_models')#required=True)
    parser.add_argument("--network", default= '2d')#required=True)
    parser.add_argument("--max_epoch", default= 1)#required=True)
    parser.add_argument("--i_lr", default= 1e-2)#required=True)
    parser.add_argument("-d","--deterministic", default=True) #,action='store_true')#required=True)
    parser.add_argument("-t", "--task", default= 'Task500_ISLES_ad')#required=True)
    parser.add_argument("-f", "--fold", default= '0')#required=True)
    parser.add_argument("-c", "--continue_training", default=False, required=False)
    parser.add_argument("-vd", "--val_data", default=[], required=False)
    parser.add_argument("-g", "--gpus", default="0") #"0,1,2,3" "0"
    parser.add_argument("-p", "--pretrained_weights", default=None) #"nnUNet_trained_models/nnUNet/3d_fullres/Task505_BRATS/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model",
    #fixed parmeters
    parser.add_argument("--network_trainer", default='nnUNetTrainerV2') 
    parser.add_argument("--plans_identifier", default='nnUNetPlansv2.1') 
    parser.add_argument("--decompress_data", default=True) 
    parser.add_argument("--batch_dice", default=False) 
    return parser

def test_setting():
    #my_output_identifier = "nnUNet"
  
    default_data_identifier = 'nnUNetData_plans_v2.1'

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', default = "/data/1_nnunet_refactoring/nnUNet_raw_data_base/nnUNet_raw_data/Task500_ISLES_ad/imagesTs")#, required=True)
    parser.add_argument('-o', "--output_folder", default = "/data/1_nnunet_refactoring/nnUNet_raw_data_base/nnUNet_raw_data/Task500_ISLES_ad/outputsTs",
                        required=False, help="folder for saving predictions")
    parser.add_argument('-t', '--task', help='task name or task ID, required.',
                        default='Task500_ISLES_ad')#, required=True)
    parser.add_argument('-tr', '--network_trainer',required=False,default='nnUNetTrainerV2')
    parser.add_argument('-ctr', '--cascade_trainer_class_name',
                        help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade."
                             "Default is %s" % "nnUNetTrainerV2CascadeFullRes", required=False,
                        default="nnUNetTrainerV2CascadeFullRes")

    parser.add_argument('-m', '--model', help="2d, 3d_fullres. Default: 3d_fullres",default="2d", required=False)

    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',
                        default= "nnUNetPlansv2.1", required=False)

    parser.add_argument('-f', '--fold', nargs='+', default='None',
                        help="folds to use for prediction. Default is None which means that folds will be detected "
                             "automatically in the model output folder")

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

    parser.add_argument("--num_parts", type=int, required=False, default=1,
                        help="Used to parallelize the prediction of "
                             "the folder over several GPUs. If you "
                             "want to use n GPUs to predict this "
                             "folder you need to run this command "
                             "n times with --part_id=0, ... n-1 and "
                             "--num_parts=n (each with a different "
                             "GPU (via "
                             "CUDA_VISIBLE_DEVICES=X)")

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
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False, help="can be None, False or True. "
                                                                                       "Do not touch.")
    parser.add_argument("--step_size", type=float, default=0.5, required=False, help="don't touch")
    parser.add_argument('-chk',
                        help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False,
                        help='Predictions are done with mixed precision by default. This improves speed and reduces '
                             'the required vram. If you want to disable mixed precision you can set this flag. Note '
                             'that this is not recommended (mixed precision is ~2x faster!)')
    parser.add_argument('--network_training_output_dir', default = "/data/1_nnunet_refactoring/nnUNet_refact_v1/nnUNet_trained_models")
    
    return parser