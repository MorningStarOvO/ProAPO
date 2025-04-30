# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from tqdm import tqdm  

import torch 
import clip 

from einops import rearrange, repeat

from utils.util_model import accuracy, get_fitness_score
from utils.util import save_json, sorted_dict, sorted_dict_by_key
from models.description.DCLIP_util import make_descriptor_sentence
from models.APO_score_func.get_score_based_ACC import get_acc_scores


# ==================== Functions ==================== #

def get_class_specific_prompt_to_category(args, prompt_dict, group_list, category_select_dict, prompt_to_category_dict):
    
    description_flag = 0
    if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT"]:
        path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 

        if not os.path.exists(path_description):
            print(f"{path_description} not exists !")
            sys.exit()

        with open(path_description, 'r') as f:
            data_description = json.load(f)

        description_flag = 1

    path_label_synonym = os.path.join("data/dataset", args.dataset.lower(), "category_synonym", f"{args.category_synonym_mode}.json")
    with open(path_label_synonym, 'r') as f:
        data_label_synonym = json.load(f)

    for temp_category in category_select_dict:
        if category_select_dict[temp_category] or args.description_init_flag:
            temp_prompt_list = prompt_dict[temp_category]
            for temp_prompt in temp_prompt_list:
                if temp_prompt not in prompt_to_category_dict:
                    prompt_to_category_dict[temp_prompt] = temp_category

    for temp_category in group_list:

        temp_label_synonym = data_label_synonym[temp_category]
        temp_prompt_list = prompt_dict[temp_category].copy()

        for temp_prompt in temp_prompt_list:
            for temp_synonym in temp_label_synonym:
                prompt_to_category_dict[temp_prompt.replace(temp_category.lower(), temp_synonym.lower())] = temp_category

    if description_flag:
        for temp_category in group_list:

            temp_description_list = data_description[temp_category]

            if args.mode_prompt_class_specific == "DCLIP":
                temp_prompt_list_new = []
                for temp_prompt in temp_prompt_list:
                    
                    if "which" in temp_prompt:
                        continue

                    for temp_description in temp_description_list:
                        descriptor = make_descriptor_sentence(temp_description)
                        temp_prompt_with_description = f"{temp_prompt.replace('.', '')}, {descriptor}."

                        if temp_prompt_with_description.lower() not in temp_prompt_list_new:
                            temp_prompt_list_new.append(temp_prompt_with_description.lower())
                
                temp_description_list = temp_prompt_list_new.copy()

            for temp_description in temp_description_list:
                if temp_description.lower() not in prompt_to_category_dict:
                    prompt_to_category_dict[temp_description.lower()] = temp_category

            
    prompt_to_category_dict_new = {}
    for temp_prompt in prompt_to_category_dict:
        prompt_to_category_dict_new[temp_prompt.lower()] = prompt_to_category_dict[temp_prompt]

    return prompt_to_category_dict_new



def build_class_specific_prompt_memory(args, model, tokenizer, group_list, category_to_label_dict, prompt_to_category_dict, baseline_result,
                                        prompt_dict, text_features, train_image_features_array, train_labels_array, 
                                        threshold_worst_decrease, prompt_delete_all_list):
    
    prompt_to_id_dict = {}
    idx_prompt = 0

    memory_prompt_dict = {}
    memory_prompt_dict["baseline"] = baseline_result
    memory_prompt_dict["delete_prompt"] = prompt_delete_all_list.copy()

    class_specific_prompt_list = []
    for idx, temp_prompt in enumerate(tqdm(prompt_to_category_dict)):

        temp_category = prompt_to_category_dict[temp_prompt]
        temp_baseline_prompt = prompt_dict[temp_category]

        if temp_prompt not in prompt_to_id_dict:
            prompt_to_id_dict[temp_prompt] = idx_prompt 
            idx_prompt += 1

        if temp_prompt in temp_baseline_prompt:
            continue
        if temp_prompt in prompt_delete_all_list:
            continue
        elif prompt_to_category_dict[temp_prompt] not in group_list:
            continue
        else:
            temp_text_features = text_features.clone().detach()
            temp_prompt_all = temp_baseline_prompt.copy()
            temp_prompt_all.append(temp_prompt)

            with torch.no_grad():
                temp_prompt_all = tokenizer(temp_prompt_all).cuda()

                temp_prompt_all = model.encode_text(temp_prompt_all)
                temp_prompt_all /= temp_prompt_all.norm(dim=-1, keepdim=True)

                temp_prompt_all = temp_prompt_all.mean(dim=0)
                temp_prompt_all /= temp_prompt_all.norm(dim=-1, keepdim=True)
                temp_prompt_all = temp_prompt_all.unsqueeze(0)

            temp_text_features[category_to_label_dict[temp_category], :] = temp_prompt_all

            avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(train_image_features_array, train_labels_array, temp_text_features, model, return_detail_flag=True, analysis_flag=True)

            
            if baseline_result["top1"] - acc_top1 >= threshold_worst_decrease: # (baseline_result["top1"] == acc_top1 and baseline_result["top5"] == acc_top5)
                memory_prompt_dict["delete_prompt"].append(temp_prompt)
                # del prompt_to_id_dict[temp_prompt]
            else:
                memory_prompt_dict[temp_prompt] = {}
                # memory_prompt_dict[temp_prompt]["avg acc"] = avg_acc
                # memory_prompt_dict[temp_prompt]["avg acc-5"] = avg_acc_5
                memory_prompt_dict[temp_prompt]["top1"] = acc_top1
                memory_prompt_dict[temp_prompt]["top5"] = acc_top5
                memory_prompt_dict[temp_prompt]["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
                # memory_prompt_dict[temp_prompt]["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
                memory_prompt_dict[temp_prompt]["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
                # memory_prompt_dict[temp_prompt]["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
                memory_prompt_dict[temp_prompt]["fitness_score"] = get_fitness_score(args, memory_prompt_dict[temp_prompt])


                class_specific_prompt_list.append(temp_prompt)

    print(f"\n\tBefore: {len(class_specific_prompt_list)}")
            
    len_group = len(group_list)
    num_max_prompt = int(len_group * args.num_max_dependent_prompt)

    if len(class_specific_prompt_list) > num_max_prompt and num_max_prompt != 0:
        class_specific_prompt_list_new = []  

        memory_prompt_dict_sorted = {}
        for temp_key in memory_prompt_dict:
            if temp_key in ["baseline", "delete_prompt"]:
                continue 

            memory_prompt_dict_sorted[temp_key] = memory_prompt_dict[temp_key]

        memory_prompt_dict_sorted = sorted_dict_by_key(memory_prompt_dict_sorted, key="fitness_score")

        for idx, temp_prompt in enumerate(memory_prompt_dict_sorted):
            if temp_prompt in class_specific_prompt_list:
                class_specific_prompt_list_new.append(temp_prompt)
            
            if len(class_specific_prompt_list_new) == args.num_max_dependent_prompt:
                break  

        class_specific_prompt_list = class_specific_prompt_list_new.copy()


    for temp_prompt in class_specific_prompt_list:
        if temp_prompt not in prompt_to_id_dict:
            prompt_to_id_dict[temp_prompt] = idx_prompt 
            idx_prompt += 1

    torch.cuda.empty_cache()

    print(f"\n\tafter: {len(class_specific_prompt_list)}")

    return memory_prompt_dict, class_specific_prompt_list, prompt_to_id_dict

