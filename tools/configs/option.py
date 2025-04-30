# ==================== Import Packages ==================== #
import argparse 


from pprint import pprint


import numpy as np 
import os 
import json 
import yaml 

import sys 


# ==================== Functions ==================== #
def parse_opt():

    parser = argparse.ArgumentParser(description='Training ProAPO!')

    parser.add_argument('--dataset', default='cub', help='dataset name')
    parser.add_argument('--data_root', default='data/CUB2002011/', help='path to dataset')

    # ----- model setting  ----- #  
    parser.add_argument('--model', default='clip_ViT-B/32', choices=["clip_RN50", "clip_RN101", 
                                                                     "clip_ViT-B/32", "clip_ViT-B/16", "clip_ViT-L/14", "clip_ViT-L/14@336px", 
                                                                     "openclip_laion2b-ViT-B-32", "openclip_laion2b-ViT-B-16", "openclip_laion2b-ViT-L-14", 
                                                                     "EVA02_B-16", "EVA02_L-14",
                                                                     "SigLIP_ViT-B-16", "SigLIP_ViT-L-16-256", "SigLIP_ViT-SO400M-14"])
    
    parser.add_argument('--source_model', default='clip_ViT-B/32', choices=["clip_RN50", "clip_RN101", 
                                                                            "clip_ViT-B/32", "clip_ViT-B/16", "clip_ViT-L/14", "clip_ViT-L/14@336px", 
                                                                            "openclip_laion2b-ViT-B-32", "openclip_laion2b-ViT-B-16", "openclip_laion2b-ViT-L-14", 
                                                                            "EVA02_B-16", "EVA02_L-14",
                                                                            "SigLIP_ViT-B-16", "SigLIP_ViT-L-16-256", "SigLIP_ViT-SO400M-14"])
    

    parser.add_argument('--name_hyper_config', default='config_num_shots_1', help='')

    parser.add_argument('--model_mode', default='train', choices=["train", "upper_bound", "zero_shot"], help='')
    parser.add_argument('--fitness_analysis_flag', action='store_true', default=False)
    parser.add_argument('--hyper_search_flag', action='store_true', default=False)
    parser.add_argument('--unable_same_template_flag', action='store_true', default=False)
    parser.add_argument('--unable_synonym_label_flag', action='store_true', default=False)
    parser.add_argument('--unable_category_important_score_flag', action='store_true', default=False)
    parser.add_argument('--delete_category_base_important_score_flag', action='store_true', default=False)
    parser.add_argument('--description_init_flag', action='store_true', default=False)
    parser.add_argument('--description_sample_flag', action='store_true', default=False)
    parser.add_argument('--black_box_flag', action='store_true', default=False)


    # ablation mode
    parser.add_argument('--ablation_mode', default='full', choices=["full", "ablation_score_only_CE", "ablation_score_only_ACC", 
                                                                    "ablation_GEN_add", "ablation_GEN_add_del", "ablation_GEN_del_rep", "ablation_GEN_add_delete_rep", 
                                                                    "ablation_EVO_cross", "ablation_EVO_mut", 
                                                                    "ablation_random_class_specific_prompt", 
                                                                    "ablation_random_selected_group", "ablation_sample_perform_good_group", "ablation_with_all_class_in_one_group"], help='')
    
    
    parser.add_argument('--num_description_sample_random', default=16, type=int, help='')
    parser.add_argument('--num_description_sample_same', default=16, type=int, help='')
    
    # =======================================================
    #                       prompt
    # =======================================================

    
    parser.add_argument('--mode_prompt_model_agnostic', default='clip', choices=["json_list", "json", "class_name", "clip", "mode_single_template",
                                                                                 "defilip_6", "filip", "imagenet_80", "imagenet_select", 
                                                                                 "defilip_6_suffix", "imagenet_80_suffix", "imagenet_select_suffix", "pre_defined_config"]) 
    parser.add_argument('--path_prompt_model_agnostic', default=None)
    parser.add_argument('--a_photo_of_a_flag', action='store_true', default=False)

    parser.add_argument('--add_dataset_species_flag', action='store_true', default=False)
    parser.add_argument('--path_dataset_species', default="data/prompt_template_task_specific/dataset_type_list.json")
    
    parser.add_argument('--path_prompt_synonym_label', default=None)
    
    parser.add_argument('--mode_prompt_class_specific', default=None, choices=[None, "DCLIP", "AWT", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "definition", "pre_defined_config"]) 
    parser.add_argument('--path_description', default=None)

    parser.add_argument('--full_prompt_flag', action='store_true', default=False)

    # =======================================================
    #                     APO
    # =======================================================
    parser.add_argument('--APO_mode', default='agnostic', choices=['agnostic', 'dependent', "agnostic_and_dependent"])

    parser.add_argument('--prompt_initial_mode', default='sample_manual', choices=['default', 'sample_manual', 'select_top'])

    parser.add_argument('--prompt_sample_mode', default='mean', choices=['mean', "weighted"])
    parser.add_argument('--weighted_mode', default='linear', choices=['linear', "exp"])
    parser.add_argument('--weighted_momentum', type=float, default=0.9)
    parser.add_argument('--threshold_epsilon_linear', default=1, type=float, help='')
    parser.add_argument('--threshold_epsilon_exp', default=1, type=float, help='')

    parser.add_argument('--validate_set_mode', default='fewshot', choices=['fewshot'])

    parser.add_argument('--epochs', default=8, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs_inner', default=4, type=int, metavar='N', help='')
    parser.add_argument('--epochs_outer', default=3, type=int, metavar='N', help='')


    parser.add_argument('--num_top', default=4, type=int, help='')
    parser.add_argument('--num_dependent_top', default=4, type=int, help='')
    

    parser.add_argument('--num_max_prompt', default=0, type=int, help='')
    parser.add_argument('--num_max_dependent_prompt', default=0, type=int, help='')
    
    parser.add_argument('--num_ATO_generate', default=0, type=int, help='')
    parser.add_argument('--num_add', default=8, type=int, help='')
    parser.add_argument('--num_remove', default=8, type=int, help='')
    parser.add_argument('--num_replace', default=8, type=int, help='')
    parser.add_argument('--num_mutation', default=8, type=int, help='')
    parser.add_argument('--num_crossover', default=16, type=int, help='')
    
    parser.add_argument('--num_ProAPO_generate', default=0, type=int, help='')
    parser.add_argument('--num_dependent_add', default=8, type=int, help='')
    parser.add_argument('--num_dependent_remove', default=8, type=int, help='')
    parser.add_argument('--num_dependent_replace', default=8, type=int, help='')
    parser.add_argument('--num_dependent_mutation', default=4, type=int, help='')
    parser.add_argument('--num_dependent_crossover', default=16, type=int, help='')


    parser.add_argument('--add_mode', default='add_one', choices=['add_one', "add_more"]) 
    parser.add_argument('--crossover_mode', default='slow', choices=['slow', "quick"]) 

    parser.add_argument('--threshold_worst_decrease', default=0, type=float, help='')
    parser.add_argument('--default_threshold_worst_decrease', default=2, type=int, help='')

    parser.add_argument('--evaluate_mode', default='top', choices=['top', "top5_mean"]) 
    parser.add_argument('--score_top_1_rate', default=0.1, type=float, help='')
    parser.add_argument('--score_top_5_rate', default=0.1, type=float, help='')
    parser.add_argument('--score_inter_class_score_CE', default=10, type=float, help='')
    parser.add_argument('--score_redundant_score_l2_norm', default=0.05, type=float, help='')
    # parser.add_argument('--score_redundant_score_most_similar', default=0, type=float, help='')

    
    parser.add_argument('--path_best_agnostic_prompt', default=None, type=str, help='')

    parser.add_argument('--threshold_confuse', default=4, type=int, help='')
    parser.add_argument('--threshold_group_mode', default='overall', choices=['under_baseline', "overall"])
    parser.add_argument('--only_performance_init_flag', action='store_true', default=False)
    parser.add_argument('--not_consider_performance_flag', action='store_true', default=False)

    parser.add_argument('--num_max_search_group', default=4, type=int, help='')
    parser.add_argument('--num_consider_rate', default=0.2, type=float, help='')
    parser.add_argument('--num_bad_delete_rate', default=0, type=float, help='')
    parser.add_argument('--num_K_means_max_rate', default=0.2, type=float, help='')

    parser.add_argument('--category_synonym_mode', default='gpt-4-turbo-2024-04-09', choices=['gpt-4-0613', 'gpt-4-turbo-2024-04-09'])

    parser.add_argument('--memory_init_flag', action='store_false', default=True)
    parser.add_argument('--memory_image_init_flag', action='store_false', default=True)

    parser.add_argument('--delete_max_similar_score', default=0.98, type=float, help='')

    parser.add_argument('--num_consider_category_each_group', default=0, type=int, help='')


    parser.add_argument('--update_record_in_group_end_mode', default='use_top1', choices=['update_same', 'unchanged', 'use_top1'])

    # =======================================================
    #                 Few-Shot Settings
    # =======================================================
    parser.add_argument('--num_shots', default=1, type=int, help='')

    
    # =======================================================
    #                    Others
    # =======================================================
    
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--batch_size_text', default=2048, type=int, metavar='N')
    
    
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--seed_data', default=42, type=int, help='seed for data')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optim', default='adam', choices=['adamw', 'adam'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8) 
    

    parser.add_argument('--task_name', default="default", type=str, help='task name')
    parser.add_argument('--model_name', default="default", type=str, help='model name')
    parser.add_argument('--checkpoint_path', default='checkpoint/ProAPO', type=str, metavar='PATH',
                        help='path to save output results')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug mode')
    parser.add_argument('--quick_run_flag', dest='quick_run_flag', action='store_true', help='debug mode')
        

    args = parser.parse_args() 
    

    if args.threshold_group_mode == "overall":
        args.threshold_group_acc = 100

    with open(os.path.join("data/dataset", args.dataset.lower(), "label_to_category_name.json"), 'r') as f:
        label_to_category_name = json.load(f)
    if args.num_consider_rate > 0:
        # args.max_num_consider = int(len(label_to_category_name) * args.num_consider_rate)
        args.max_num_consider = int(np.log(len(label_to_category_name)) * 10)

        if args.max_num_consider > len(label_to_category_name):
            args.max_num_consider = len(label_to_category_name)
        

        if args.num_bad_delete_rate == 0:
            args.num_bad_delete_rate = (1 - args.num_consider_rate) * 0.5
    else:
        args.max_num_consider = 0

    if args.threshold_worst_decrease == 0:
        args.threshold_worst_decrease = (100 * args.default_threshold_worst_decrease) / len(label_to_category_name) + 0.1
        print("\nmodified threshold_worst_decrease: ", args.threshold_worst_decrease)

    if args.num_consider_category_each_group == 0:
        args.num_consider_category_each_group = int(np.log(len(label_to_category_name)) * 2.5)
    
    root_dir = os.getcwd().split("ProAPO")[0]
    args.data_root = os.path.join(root_dir, args.data_root)
    args.checkpoint_path = os.path.join(root_dir, args.checkpoint_path)
    print("\nmodified path: ", args.data_root)

    return args 
