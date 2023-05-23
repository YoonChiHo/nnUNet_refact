
from lib.train.loss import DC_and_CE_loss #가능하면 세팅가져와 사용할수있게 간단히
from lib.train.generic_UNet import Generic_UNet, InitWeights_He, softmax_helper #제일 수정해야할것, 일단 보류
from lib.train.dataloader import load_dataset, setup_DA_params, do_split, DataLoader3D, DataLoader2D, unpack_dataset, get_moreDA_augmentation
from lib.preprocess.preprocessing import GenericPreprocessor
from lib.utils import print_to_log_file, to_cuda, sum_tensor, maybe_to_torch

from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle, save_json

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from torch import nn
from collections import OrderedDict
from _warnings import warn
from time import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l
    
class nnUNetTrainer_simple(object):
    def __init__(self, plans_file, fold, output_folder, dataset_directory,#required
                 deterministic=True, i_lr=1e-2, val_data=None, max_epoch=1000,    #editable
                 fp16=True, unpack_data=True, stage=0, batch_dice=False): #default settings

        #network_trainer setting
        self.fp16 = fp16

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = max_epoch#1000
        self.num_batches_per_epoch = 250
        self.num_val_batches_per_epoch = 50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        ################# Settings for saving checkpoints ##################################
        self.save_every = 10
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

        #nnUNetTrainer setting
        self.unpack_data = unpack_data
        self.init_args = (plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                          deterministic, fp16)
        # set through arguments from init
        self.stage = stage
        self.experiment_name = self.__class__.__name__
        self.plans_file = plans_file
        self.output_folder = output_folder
        self.dataset_directory = dataset_directory
        self.output_folder_base = self.output_folder

        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        self.gt_niftis_folder = os.path.join(self.dataset_directory, "gt_segmentations")
        self.folder_with_preprocessed_data = None

        self.loss = DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.inference_pad_border_mode = "constant"
        self.inference_pad_kwargs = {'constant_values': 0}

        self.net_num_pool_op_kernel_sizes = None

        self.fold = fold #가능하면 5개 fold 한꺼번에 학습하는 방법도세팅해두기
        self.update_fold(fold)
        self.pad_all_sides = None

        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        self.initial_lr = i_lr#3e-4
        self.weight_decay = 3e-5

        self.oversample_foreground_percent = 0.33

        self.regions_class_order = None
        self.pin_memory = True
        self.val_data = val_data

    def update_fold(self, fold):
        """
        used to swap between folds for inference (ensemble of models from cross-validation)
        DO NOT USE DURING TRAINING AS THIS WILL NOT UPDATE THE DATASET SPLIT AND THE DATA AUGMENTATION GENERATORS
        :param fold:
        :return:
        """
        if fold is not None:
            if isinstance(fold, str):
                assert fold == "all", "if self.fold is a string then it must be \'all\'"
                if self.output_folder.endswith("%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = os.path.join(self.output_folder, "%s" % str(fold))
            else:
                if self.output_folder.endswith("fold_%s" % str(self.fold)):
                    self.output_folder = self.output_folder_base
                self.output_folder = os.path.join(self.output_folder, "fold_%s" % str(fold))
            self.fold = fold

    def initialize(self, training=True): #force_load_plans 불필요
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :return:
        """
        os.makedirs(self.output_folder, exist_ok=True)

        self.plans = load_pickle(self.plans_file) 

        self.process_plans(self.plans)
        # Setup Data Augmentation Parameters
        self = setup_DA_params(self)

        # 위에서 불필요한 value는 return할 필요없음

        ################# Here we wrap the loss for deep supervision ############
        # we need to know the number of outputs of the network
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        # now wrap the loss
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
        ################# END ###################

        self.folder_with_preprocessed_data = os.path.join(self.dataset_directory, self.plans['data_identifier'] +
                                                    "_stage%d" % self.stage)
        if training:
            # Get basic generators (data loader)
            #self.dl_tr, self.dl_val = self.get_basic_generators()
            self.dataset = load_dataset(self.folder_with_preprocessed_data)

            self = do_split(self)

            if self.threeD:
                self.dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    False, oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                self.dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            else:
                self.dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                self.dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            
            if self.unpack_data:
                print("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                print("done")
            else:
                print(
                    "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                    "will wait all winter for your model to finish!")

            self.tr_gen, self.val_gen = get_moreDA_augmentation(
                self.dl_tr, self.dl_val,
                self.data_aug_params[
                    'patch_size_for_spatialtransform'],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                use_nondetMultiThreadedAugmenter=False
            )
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                    also_print_to_console=False)
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                    also_print_to_console=False)
        else:
            pass

        # Initialize Network
        #self.initialize_network()
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        if training: do_ds = True
        else: do_ds = False
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, do_ds, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


        # Initialize Opti and Scheduler
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

        self.was_initialized = True

    def process_plans(self, plans):
        # Plans 값 불러와서 모델세팅 (필요시 직접 값 변경할 수 있도록 세팅)
        stage_plans = plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

    def run_training(self):
        #update lr
        
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        ep = self.epoch #현재 에포크
        self.optimizer.param_groups[0]['lr'] = self.initial_lr * (1 - ep / self.max_num_epochs)**0.9 
        self.log_file = print_to_log_file(self.log_file,self.output_folder,"lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True

        #ret = super().run_training()

        self.save_debug_information()
        #network_trainer의 run_training
        #super(nnUNetTrainer, self).run_training()

        if not torch.cuda.is_available():
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.fp16:
            self.amp_grad_scaler = GradScaler()

        #maybe_mkdir_p(self.output_folder)       
        os.makedirs(self.output_folder, exist_ok=True)

        #self.plot_network_architecture() #필요없을지도

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        # if not self.was_initialized: #True
        #     self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            # if self.use_progress_bar: #일단 불필요
            #     with trange(self.num_batches_per_epoch) as tbar:
            #         for b in tbar:
            #             tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

            #             l = self.run_iteration(self.tr_gen, True)

            #             tbar.set_postfix(loss=l)
            #             train_losses_epoch.append(l)
            # else:
            for _ in range(self.num_batches_per_epoch):
                # Run Iteration Code
                l = self.run_iteration(self.tr_gen, True)
                train_losses_epoch.append(l) # 

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"train loss : %.4f" % self.all_tr_losses[-1])

            # 여기까지 1 epoch 끝
            # 아래부터 Evaluation
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.log_file = print_to_log_file(self.log_file,self.output_folder,"validation loss: %.4f" % self.all_val_losses[-1])

                # if self.also_val_in_tr_mode: #False
                #     self.network.train()
                #     # validation with train=True
                #     val_losses = []
                #     for b in range(self.num_val_batches_per_epoch):
                #         l = self.run_iteration(self.val_gen, False)
                #         val_losses.append(l)
                #     self.all_val_losses_tr_mode.append(np.mean(val_losses))
                #     self.log_file = print_to_log_file(self.log_file,self.output_folder,"validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            #self.update_train_loss_MA()  # needed for lr scheduler and stopping of training
            if self.train_loss_MA is None:
                self.train_loss_MA = self.all_tr_losses[-1]
            else:
                self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                    self.all_tr_losses[-1]
            
            #continue_training = self.on_epoch_end() #수정이 필요함
            self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
            # metrics
            ## 230505check필요, 다시 훈련돌려보기
            self.plot_progress()
            self.maybe_update_lr()
            self.maybe_save_checkpoint()
            self.update_eval_criterion_MA()
            continue_training = self.manage_patience()

            continue_training = self.epoch < self.max_num_epochs

            # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
            # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
            if self.epoch == 100:
                if self.all_val_eval_metrics[-1] == 0:
                    self.optimizer.param_groups[0]["momentum"] = 0.95
                    self.network.apply(InitWeights_He(1e-2))
                    self.log_file = print_to_log_file(self.log_file,self.output_folder,"At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                        "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                        "sometimes causes issues such as this one. Momentum has now been reduced to "
                                        "0.95 and network weights have been reinitialized")

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(os.path.join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if os.path.isfile(os.path.join(self.output_folder, "model_latest.model")):
            os.remove(os.path.join(self.output_folder, "model_latest.model"))
        if os.path.isfile(os.path.join(self.output_folder, "model_latest.model.pkl")):
            os.remove(os.path.join(self.output_folder, "model_latest.model.pkl"))


        self.network.do_ds = ds

        #return ret #1?
    
    def run_online_evaluation(self, output, target):
        # Select Biggist Resolution
        target = target[0]
        output = output[0]
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.log_file = print_to_log_file(self.log_file,self.output_folder,"Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.log_file = print_to_log_file(self.log_file,self.output_folder,"(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):


        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        # else:
        #     output = self.network(data)
        #     del data
        #     l = self.loss(output, target)

        #     if do_backprop:
        #         l.backward()
        #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #         self.optimizer.step()

        if run_online_evaluation: #False
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
    
    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                if self.save_best_checkpoint: self.save_checkpoint(os.path.join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
            else:
                pass
            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    continue_training = False
            else:
                pass
        return continue_training
    
    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        #valid = list((SegmentationNetwork, nn.DataParallel))
        #assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ##TODO sa
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret
    
    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.log_file = print_to_log_file(self.log_file,self.output_folder,"lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def plot_network_architecture(self):
        try:
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                    transforms=None)
            else:
                g = hl.build_graph(self.network, torch.rand((1, self.num_input_channels, *self.patch_size)),
                                    transforms=None)
            g.save(os.path.join(self.output_folder, "network_architecture.pdf"))
            del g
        except Exception as e:
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"Unable to plot network architecture:")
            self.log_file = print_to_log_file(self.log_file,self.output_folder,e)

            self.log_file = print_to_log_file(self.log_file,self.output_folder,"\nprinting the network instead:\n")
            self.log_file = print_to_log_file(self.log_file,self.output_folder,self.network)
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(os.path.join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(os.path.join(self.output_folder, "model_latest.model"))
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]
                
    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.log_file = print_to_log_file(self.log_file,self.output_folder,"saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.log_file = print_to_log_file(self.log_file,self.output_folder,"done, saving took %.2f seconds" % (time() - start_time))

        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        save_pickle(info, fname + ".pkl")

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, os.path.join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, os.path.join(self.output_folder_base, "plans.pkl"))


    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(os.path.join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.log_file = print_to_log_file(self.log_file,self.output_folder,"failed to plot: ", sys.exc_info())

    def preprocess_patient(self, input_files):
            """
            Used to predict new unseen data. Not used for the preprocessing of the training/test data
            :param input_files:
            :return:
            """
            # from nnunet.training.model_restore import recursive_find_python_class
            # preprocessor_name = self.plans.get('preprocessor_name')
            # if preprocessor_name is None:
            #     if self.threeD:
            #         preprocessor_name = "GenericPreprocessor"
            #     else:
            #         preprocessor_name = "PreprocessorFor2D"

            # print("using preprocessor", preprocessor_name)
            # preprocessor_class = recursive_find_python_class([os.path.join(nnunet.__path__[0], "preprocessing")],
            #                                                 preprocessor_name,
            #                                                 current_module="nnunet.preprocessing")
            # assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
            #                                     preprocessor_name
            preprocessor = GenericPreprocessor(self.normalization_schemes, self.use_mask_for_norm,
                                            self.transpose_forward, self.intensity_properties)

            d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                                self.plans['plans_per_stage'][self.stage][
                                                                    'current_spacing'])
            return d, s, properties
    
    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            #self._maybe_init_amp()
            if self.fp16:
                self.amp_grad_scaler = GradScaler()
            if train:
                if 'amp_grad_scaler' in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            #if issubclass(self.lr_scheduler.__class__, _LRScheduler):
            self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        #self._maybe_init_amp()
        if self.fp16:
            self.amp_grad_scaler = GradScaler()
        

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['state_dict']

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        # remove module. prefix from DDP models
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if ('conv_blocks' in key):
            if (key in pretrained_dict) and (model_dict[key].shape == pretrained_dict[key].shape):
                continue
            else:
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("################### Loading pretrained weights from file ", fname, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError("Pretrained weights are not compatible with the current network architecture")

