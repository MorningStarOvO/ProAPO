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
from models.util_prompt import get_prompt_from_template_and_description

# ==================== Functions ==================== #

def get_class_specific_prompt_to_category(args, prompt_template_key_dict, class_name_list, best_prompt_agnostic, group_list, category_select_dict, prompt_to_category_dict):

    # ----- read description ----- # 
    description_flag = 0
    if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
        path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 

        if not os.path.exists(path_description):
            print(f"{path_description} not exists !")
            sys.exit()

        with open(path_description, 'r') as f:
            data_description = json.load(f)

        description_flag = 1

    # ----- read label synonym ----- # 
    if not args.unable_synonym_label_flag:

        if args.path_prompt_synonym_label is not None:
            path_label_synonym = os.path.join(args.path_best_prompt_save, f"{args.path_prompt_synonym_label}.json")
        else:
            path_label_synonym = os.path.join("data/dataset", args.dataset.lower(), "category_synonym", f"{args.category_synonym_mode}.json")

        with open(path_label_synonym, 'r') as f:
            data_label_synonym = json.load(f)

    for temp_category in category_select_dict:
        if category_select_dict[temp_category] or args.description_init_flag:
            temp_prompt_list = prompt_template_key_dict[temp_category]
            for temp_prompt in temp_prompt_list:
                if temp_prompt not in prompt_to_category_dict:
                    prompt_to_category_dict[temp_prompt] = temp_category

    for temp_category in group_list:
        if args.unable_synonym_label_flag:
            temp_label_synonym = [temp_category]
        else:
            temp_label_synonym = data_label_synonym[temp_category]

            if temp_category.lower() not in temp_label_synonym:
                temp_label_synonym.append(temp_category.lower())

  
        for temp_synonym in temp_label_synonym:
            if temp_synonym not in prompt_to_category_dict:
                prompt_to_category_dict[temp_synonym] = temp_category

            if description_flag:
            
                temp_description_list = data_description[temp_category]

                if args.mode_prompt_class_specific == "DCLIP":
                    temp_prompt_list_new = []

                    for temp_description in temp_description_list:
                        descriptor = make_descriptor_sentence(temp_description)
                        
                        if descriptor not in temp_prompt_list_new:
                            temp_prompt_list_new.append(descriptor)
                    
                    temp_description_list = temp_prompt_list_new.copy()

                for temp_description in temp_description_list:
                    temp_description_new = temp_description.lower().replace(temp_category, temp_synonym)
                    temp_combine_prompt = f"{temp_synonym}**+*+**{temp_description_new}"

                    if temp_combine_prompt not in prompt_to_category_dict:
                        prompt_to_category_dict[temp_combine_prompt] = temp_category


    return prompt_to_category_dict


def build_class_specific_prompt_memory(args, model, tokenizer, group_list, category_to_label_dict, prompt_to_category_dict, baseline_result,
                                        prompt_template_key_dict, class_name_list, best_prompt_agnostic, text_features, train_image_features_array, train_labels_array, 
                                        threshold_worst_decrease, prompt_delete_all_list, category_group_baseline_features):
    """build class-specific prompt library"""

    prompt_to_id_dict = {}
    idx_prompt = 0

    memory_prompt_dict = {}
    memory_prompt_dict["baseline"] = baseline_result
    memory_prompt_dict["delete_prompt"] = prompt_delete_all_list.copy()

    class_specific_prompt_list = []
    for idx, temp_prompt in enumerate(tqdm(prompt_to_category_dict)):

        temp_category = prompt_to_category_dict[temp_prompt]
        temp_baseline_prompt = prompt_template_key_dict[temp_category]

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

            temp_prompt_all = [temp_prompt]

            temp_prompt_all = get_prompt_from_template_and_description(best_prompt_agnostic, temp_prompt_all)
            
            # ----- Text embedding ----- # 
            with torch.no_grad():
                temp_prompt_all = tokenizer(temp_prompt_all).cuda()

                temp_prompt_all = model.encode_text(temp_prompt_all)

                temp_prompt_all = torch.cat((category_group_baseline_features[temp_category], temp_prompt_all))

                temp_prompt_all /= temp_prompt_all.norm(dim=-1, keepdim=True)

                temp_prompt_all = temp_prompt_all.mean(dim=0)
                temp_prompt_all /= temp_prompt_all.norm(dim=-1, keepdim=True)
                temp_prompt_all = temp_prompt_all.unsqueeze(0)

            temp_text_features[category_to_label_dict[temp_category], :] = temp_prompt_all

            avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(train_image_features_array, train_labels_array, temp_text_features, model, return_detail_flag=True, analysis_flag=True)


            if baseline_result["top1"] - acc_top1 >= threshold_worst_decrease: # (baseline_result["top1"] == acc_top1 and baseline_result["top5"] == acc_top5)
                memory_prompt_dict["delete_prompt"].append(temp_prompt)
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

    print(f"\n\tbefore: {len(class_specific_prompt_list)}")
            
    len_group = len(group_list)
    num_max_prompt = int(len_group * args.num_max_dependent_prompt)
    
    if num_max_prompt != 0:
        print("\nnum_max_prompt: ", num_max_prompt)

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
            
            if len(class_specific_prompt_list_new) == num_max_prompt:
                break  

        class_specific_prompt_list = class_specific_prompt_list_new.copy()

    for temp_prompt in class_specific_prompt_list:
        if temp_prompt not in prompt_to_id_dict:
            prompt_to_id_dict[temp_prompt] = idx_prompt 
            idx_prompt += 1

    
    torch.cuda.empty_cache()

    print(f"\n\tconsider nums:: {len(class_specific_prompt_list)}")

    return memory_prompt_dict, class_specific_prompt_list, prompt_to_id_dict

