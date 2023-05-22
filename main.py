import os

from options import initial_setting
from lib.trainer import nnUNetTrainer_simple, load_pretrained_weights
from lib.utils import load_pickle

#Main Parameter Settings
args = initial_setting().parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus # "0, 1, 2, 3"
if args.fold == 'all': fold = args.fold
else: fold = int(args.fold)

# pych_Network Validate
assert args.network in ['2d', '3d_fullres'], "network can only be one of the following: \'2d\',  \'3d_fullres\''"

# Set Directory
output_folder_name = os.path.join(args.checkpoints, args.network, args.task,f'{args.network_trainer}__{args.plans_identifier}')  #더 줄여도 괜찮 (network_trainer이랑 plans_identifier은 빼돋 ㅚㄹ듯)
dataset_directory = os.path.join(args.dataset, args.task)

# Load Plans File
if args.network == '2d':
    plans_file = os.path.join(args.dataset, args.task, args.plans_identifier + "_plans_2D.pkl")
else:
    plans_file = os.path.join(args.dataset, args.task, args.plans_identifier + "_plans_3D.pkl")
plans = load_pickle(plans_file)
stage = list(plans['plans_per_stage'].keys())[-1]

# 코드로 수정1
trainer = nnUNetTrainer_simple(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                        batch_dice=args.batch_dice, stage=stage, unpack_data=args.decompress_data,
                        deterministic=args.deterministic, i_lr = args.i_lr, val_data=args.val_data, max_epoch=args.max_epoch)

# 코드로 수정2
trainer.initialize()

if args.continue_training:
    # 코드로 수정3
    trainer.load_latest_checkpoint()
elif args.pretrained_weights is not None:
    # 코드로 수정4
    load_pretrained_weights(trainer.network, args.pretrained_weights) #나중에 다시 체크
else:
    pass

# 코드로 수정5
# Main Training
trainer.run_training() # Main
