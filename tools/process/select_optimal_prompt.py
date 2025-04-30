# ==================== Import Packages ==================== #
import time
import sys
import os 

import torch  
import clip 

from tqdm import tqdm  

import numpy as np 
import json 
import random
import copy
import math 
import yaml 

from models.util_prompt import init_dataset_prompt, get_prompt_dict
from models.util_get_features import get_image_features
from utils.util import sorted_dict
from process.task_specific_selection import iterative_select_task_specific_optimal_prompt
from process.class_specific_selection import iterative_select_class_specific_optimal_prompt
from process.class_specific_selection_with_same_template import iterative_select_class_specific_optimal_prompt_with_same_template


# ==================== Functions ==================== #

def iterative_select_optimal_prompt(args, model, tokenizer, class_name_list, train_loader, test_loader, threshold_worst_decrease, exp_name):
    
    # ---------- step1 ---------- # 
    print("\tget image features =>")
    model_name_str = args.model.replace("/", "_").replace("@", "_")
    if args.validate_set_mode == "fewshot":
        mode = f"{model_name_str}_seed_{args.seed_data}_num_shots_{args.num_shots}"
    test_labels_array, test_image_features_array = get_image_features(args, test_loader, model, mode=f"{model_name_str}_test")
    if args.model_mode == "upper_bound":
        train_labels_array = test_labels_array.clone().detach()
        train_image_features_array = test_image_features_array.clone().detach()
    else:
        train_labels_array, train_image_features_array = get_image_features(args, train_loader, model, mode=mode)
    
    # ---------- step2: ATO ---------- # 

    if args.path_best_agnostic_prompt is not None and args.APO_mode == "agnostic_and_dependent":
        print("\tload best template =>")
        path_best_prompt = os.path.join(args.path_best_prompt_save, f"{args.path_best_agnostic_prompt}.json") 
        with open(path_best_prompt, 'r') as f:
            best_prompt = json.load(f)
        best_prompt = best_prompt["top1"]
    elif args.APO_mode in ["agnostic", "agnostic_and_dependent"]:
        print("\tautomatic template optimization =>")

        # ----- 根据设置, 微改 args ----- # 
        if args.black_box_flag:
            args.score_inter_class_score_CE = 0
            args.score_redundant_score_l2_norm = 0
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0
        else:
            path_yaml = os.path.join("data/dataset/", args.dataset.lower(), f"config/{args.name_hyper_config}.yaml")
            if os.path.exists(path_yaml) and not args.hyper_search_flag:
                cfg = yaml.load(open(path_yaml, 'r'), Loader=yaml.Loader)
                if "agnostic" in cfg:
                    print("\n\t------ Modify the hyperparameters according to the config ^.^ ! ! ! ------")
                    args.score_inter_class_score_CE = cfg["agnostic"]["score_inter_class_score_CE"]
                    args.score_redundant_score_l2_norm = cfg["agnostic"]["score_redundant_score_l2_norm"]
                    args.score_top_1_rate = cfg["agnostic"]["score_top_1_rate"]
                    args.score_top_5_rate = cfg["agnostic"]["score_top_5_rate"]
                    
                    if "num_max_prompt" in cfg["agnostic"]:
                        args.num_max_prompt = cfg["agnostic"]["num_max_prompt"]

                    if "epochs" in cfg["agnostic"]:
                        args.epochs = cfg["agnostic"]["epochs"]

                    if "crossover_mode" in cfg["agnostic"]:
                        args.crossover_mode = cfg["agnostic"]["crossover_mode"]

                    if "prompt_sample_mode" in cfg["agnostic"]:
                        args.prompt_sample_mode = cfg["agnostic"]["prompt_sample_mode"] 

        if args.model_mode == "upper_bound":
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 0

        # ----- ablation settings ----- # 
        if args.ablation_mode == "ablation_score_only_CE":
            args.score_top_1_rate = 0
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 10
        elif args.ablation_mode == "ablation_score_only_ACC":
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 0

        if args.ablation_mode == "ablation_GEN_add":
            args.num_remove = 0
            args.num_replace = 0
            args.num_mutation = 0
            args.num_crossover = 0
        elif args.ablation_mode == "ablation_GEN_add_del":
            args.num_replace = 0
            args.num_mutation = 0
            args.num_crossover = 0
        elif args.ablation_mode == "ablation_GEN_del_rep":
            args.num_add = 0
            args.num_mutation = 0
            args.num_crossover = 0
        elif args.ablation_mode == "ablation_GEN_add_delete_rep":
            args.num_mutation = 0
            args.num_crossover = 0
        elif args.ablation_mode == "ablation_EVO_cross":
            args.num_mutation = 0
        elif args.ablation_mode == "ablation_EVO_mut":
            args.num_crossover = 0

        if args.num_ATO_generate > 0:
            args.num_add = args.num_ATO_generate
            args.num_remove = args.num_ATO_generate
            args.num_replace = args.num_ATO_generate
            args.num_mutation = args.num_ATO_generate
            args.num_crossover = args.num_ATO_generate

    
        best_prompt = iterative_select_task_specific_optimal_prompt(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, test_image_features_array, test_labels_array, threshold_worst_decrease, exp_name)
        best_prompt = best_prompt["top1"]

        if args.APO_mode == "agnostic":
            best_prompt_dict = get_prompt_dict(args, best_prompt, class_name_list)
            return best_prompt_dict, exp_name

    elif args.APO_mode == "dependent":
        print("\tinit template =>")
        best_prompt = init_dataset_prompt(args, args.dataset.lower())

    # ---------- step3: description optimization ---------- #
    if args.APO_mode == "dependent" or args.APO_mode == "agnostic_and_dependent":
        print("\tautomatic description optimization =>")
        exp_name_new = f"{exp_name}_{args.APO_mode}"

        path_yaml = os.path.join("data/dataset/", args.dataset.lower(), f"config/{args.name_hyper_config}.yaml")
        if os.path.exists(path_yaml) and not args.hyper_search_flag:
            cfg = yaml.load(open(path_yaml, 'r'), Loader=yaml.Loader)
            if "specific" in cfg:

                if not args.black_box_flag:
                    print("\n\t------ Modify the hyperparameters according to the config ^.^ ! ! ! ------")
                    args.score_inter_class_score_CE = cfg["specific"]["score_inter_class_score_CE"]
                    args.score_redundant_score_l2_norm = cfg["specific"]["score_redundant_score_l2_norm"]
                    args.score_top_1_rate = cfg["specific"]["score_top_1_rate"]
                    args.score_top_5_rate = cfg["specific"]["score_top_5_rate"]

                if "description_init_flag" in cfg["specific"]:
                    if cfg["specific"]["description_init_flag"]:
                        args.description_init_flag = True 
                        
                if "description_sample_flag" in cfg["specific"]:
                    if cfg["specific"]["description_sample_flag"]:
                        args.description_sample_flag = True
                        
                if "unable_category_important_score_flag" in cfg["specific"]:
                    if cfg["specific"]["unable_category_important_score_flag"]:
                        args.unable_category_important_score_flag = True
                        
                if "num_max_search_group" in cfg["specific"]:
                    args.num_max_search_group = cfg["specific"]["num_max_search_group"]

                if "epochs_outer" in cfg["specific"]:
                    args.epochs_outer = cfg["specific"]["epochs_outer"]

            
        if args.black_box_flag:
            args.score_inter_class_score_CE = 0
            args.score_redundant_score_l2_norm = 0
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0

        if args.dataset.lower() in ["dtd", "imagenet", "sun397", "places365", "caltech101", "euro_sat", "fgvc_aircraft"] and not args.description_sample_flag and args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
            args.description_sample_flag = True

        if args.model_mode == "upper_bound":
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 0
        
        if args.ablation_mode == "ablation_score_only_CE":
            args.score_top_1_rate = 0
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 10
        elif args.ablation_mode == "ablation_score_only_ACC":
            args.score_top_1_rate = 1
            args.score_top_5_rate = 0 
            args.score_inter_class_score_CE = 0

        if args.ablation_mode == "ablation_GEN_add":
            args.num_dependent_remove = 0
            args.num_dependent_replace = 0
            args.num_dependent_crossover = 0
            args.num_dependent_mutation = 0
        elif args.ablation_mode == "ablation_GEN_add_del":
            args.num_dependent_replace = 0
            args.num_dependent_crossover = 0
            args.num_dependent_mutation = 0
        elif args.ablation_mode == "ablation_GEN_del_rep":
            args.num_dependent_add = 0
            args.num_dependent_crossover = 0
            args.num_dependent_mutation = 0
        elif args.ablation_mode == "ablation_GEN_add_delete_rep":
            args.num_dependent_crossover = 0
            args.num_dependent_mutation = 0
        elif args.ablation_mode == "ablation_EVO_cross":
            args.num_dependent_mutation = 0
        elif args.ablation_mode == "ablation_EVO_mut":
            args.num_dependent_crossover = 0

        if args.num_ProAPO_generate > 0:
            args.num_dependent_add = args.num_ProAPO_generate
            args.num_dependent_remove = args.num_ProAPO_generate
            args.num_dependent_replace = args.num_ProAPO_generate
            args.num_dependent_mutation = args.num_ProAPO_generate
            args.num_dependent_crossover = args.num_ProAPO_generate

        if args.unable_same_template_flag:
            best_model_prompt = iterative_select_class_specific_optimal_prompt(args, model, tokenizer, best_prompt, class_name_list, train_image_features_array, train_labels_array, test_image_features_array, test_labels_array, threshold_worst_decrease, exp_name_new)
            best_model_prompt = best_model_prompt["top1"]
        else:
            best_model_prompt = iterative_select_class_specific_optimal_prompt_with_same_template(args, model, tokenizer, best_prompt, class_name_list, train_image_features_array, train_labels_array, test_image_features_array, test_labels_array, threshold_worst_decrease, exp_name_new)

        return best_model_prompt, exp_name_new
