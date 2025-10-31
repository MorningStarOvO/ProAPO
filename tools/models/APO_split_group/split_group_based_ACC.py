# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from utils.util import save_json, sorted_dict, sorted_dict_by_key
from models.util_prompt import get_prompt_dict, get_full_prompt_from_json_path
from models.util_get_features import get_text_features
from models.APO_score_func.get_score_based_ACC import get_acc_scores, get_acc_scores_and_dict_quick_and_group
from utils.util_model import get_fitness_score


# ==================== Functions ==================== #

def get_category_group_split(args, model, tokenizer, class_name_list, image_features, labels, prompt_dict):
    
    text_features = get_text_features(model, tokenizer, prompt_dict) # [category_num, feature_dim]

    avg_acc, avg_acc_5, acc_top1, acc_top5, acc_dict, category_group_split, extra_info = get_acc_scores_and_dict_quick_and_group(class_name_list, image_features, labels, text_features, model, 
                                                                                                                                 args.threshold_confuse, args.threshold_group_mode, args.max_num_consider, 
                                                                                                                                 return_detail_flag=True, analysis_flag=True)
    
    baseline_result = {}
    # baseline_result["avg acc"] = avg_acc
    # baseline_result["avg acc-5"] = avg_acc_5
    baseline_result["top1"] = acc_top1
    baseline_result["top5"] = acc_top5
    baseline_result["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
    # baseline_result["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
    baseline_result["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
    # baseline_result["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
    baseline_result["fitness_score"] = get_fitness_score(args, baseline_result)

    print("\nThe total number of current groups is: ", len(category_group_split))

    return category_group_split, text_features, baseline_result

def init_category_group_split(args, model, tokenizer, class_name_list, image_features, labels, prompt_template):
    
    if args.description_init_flag:
        print("\n\tRead description ing ...")

        if args.mode_prompt_class_specific not in [None, "definition"]:
            path_description_init = os.path.join(args.path_dataset_output, "features", f"{args.mode_prompt_class_specific}_{args.path_best_agnostic_prompt}.json")
            path_text_features_init = os.path.join(args.path_dataset_output, "features", f"{args.mode_prompt_class_specific}_{args.path_best_agnostic_prompt}.pt")
            cache_flag = 1
        else:
            cache_flag = 0
            path_description_init = None
            path_text_features_init = None

        if cache_flag and os.path.exists(path_description_init) and os.path.exists(path_text_features_init):
            with open(path_description_init, 'r') as f:
                prompt_dict = json.load(f)

            text_features = torch.load(path_text_features_init)
            text_features = text_features.cuda()
        else:

            if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
                path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 
            else:
                path_description = args.path_description

            with open(path_description, 'r') as f:
                    data_description = json.load(f)
            prompt_template_key_dict = {}
            for temp_category in class_name_list:
                temp_description_list = data_description[temp_category]
                prompt_template_key_dict[temp_category] = []
                for temp_description in temp_description_list:
                    prompt_template_key_dict[temp_category].append(f"{temp_category.lower()}**+*+**{temp_description.lower()}")

            prompt_dict = {}
            for temp_category in tqdm(class_name_list):
                prompt_dict[temp_category] = get_prompt_from_template_and_description(prompt_template, prompt_template_key_dict[temp_category])

            text_features = get_text_features(model, tokenizer, prompt_dict) 

            if cache_flag:
                save_json(prompt_dict, path_description_init)
                torch.save(text_features.clone().detach().cpu(), path_text_features_init)
            
    else:
        prompt_dict = get_prompt_dict(args, prompt_template, class_name_list)

        text_features = get_text_features(model, tokenizer, prompt_dict) # [category_num, feature_dim]\

        if not args.unable_same_template_flag:
            prompt_template_key_dict = {}
            for temp_category in class_name_list:
                prompt_template_key_dict[temp_category] = [temp_category.lower()]


    avg_acc, avg_acc_5, acc_top1, acc_top5, acc_dict, category_group_split, extra_info = get_acc_scores_and_dict_quick_and_group(class_name_list, image_features, labels, text_features, model, 
                                                                                                                                 args.threshold_confuse, args.threshold_group_mode, args.max_num_consider, 
                                                                                                                                 return_detail_flag=True, analysis_flag=True)
    

    
    baseline_result = {}
    # baseline_result["avg acc"] = avg_acc
    # baseline_result["avg acc-5"] = avg_acc_5
    baseline_result["top1"] = acc_top1
    baseline_result["top5"] = acc_top5
    baseline_result["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
    # baseline_result["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
    baseline_result["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
    # baseline_result["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
    baseline_result["fitness_score"] = get_fitness_score(args, baseline_result)

    print("\nThe total number of current groups is: ", len(category_group_split))

    if not args.unable_same_template_flag:
        return category_group_split, prompt_dict, prompt_template_key_dict, text_features, baseline_result
    else:
        return category_group_split, prompt_dict, text_features, baseline_result
 