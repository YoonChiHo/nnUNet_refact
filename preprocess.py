import os

from options import preprocess_setting
from lib.utils import load_json
from lib.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21 as planner_3d
from lib.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21 as planner_2d
#from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
import shutil

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

print("\n\n\n", args.task)
cropped_out_dir = os.path.join(args.cropped_out_dir, args.task)
preprocessing_output_dir_this_task = os.path.join(args.dataset, args.task)
#splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
#lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

# we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
dataset_json = load_json(os.path.join(cropped_out_dir, 'dataset.json'))
modalities = list(dataset_json["modality"].values())
collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
#dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=args.tf)  # this class creates the fingerprint
###_ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner


#maybe_mkdir_p(preprocessing_output_dir_this_task)
os.makedirs(preprocessing_output_dir_this_task, exist_ok=True)

shutil.copy(os.path.join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
shutil.copy(os.path.join(args.nnUNet_raw_data, args.task, "dataset.json"), preprocessing_output_dir_this_task)

threads = (args.tl, args.tf)

print("number of threads: ", threads, "\n")

if planner_3d is not None:
    if args.overwrite_plans is not None:
        assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
        exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
                                    args.overwrite_plans_identifier)
    else:
        exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
    exp_planner.plan_experiment()
    if not args.no_pp:  # double negative, yooo
        exp_planner.run_preprocessing(threads)
if planner_2d is not None:
    exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
    exp_planner.plan_experiment()
    if not args.no_pp:  # double negative, yooo
        exp_planner.run_preprocessing(threads)

