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

from utils.util import save_json, sorted_dict, sorted_dict_by_key
from models.APO_score_func.get_score_based_ACC import get_acc_scores
from models.util_prompt import get_prompt_dict, add_dataset_type_to_template, init_dataset_prompt
from models.util_get_features import get_task_specific_text_features_without_mean
from utils.util_model import get_fitness_score

# ==================== Functions ==================== #

def get_task_specific_prompt_knowledge_base(args):

    if args.mode_prompt_model_agnostic == "defilip_6":
        path_template = "data/prompt_template_task_specific/defilip_6.json"
    elif args.mode_prompt_model_agnostic == "filip":
        path_template = "data/prompt_template_task_specific/filip.json"
    elif args.mode_prompt_model_agnostic == "imagenet_80":
        path_template = "data/prompt_template_task_specific/imagenet_80.json"
    elif args.mode_prompt_model_agnostic == "imagenet_select":
        path_template = "data/prompt_template_task_specific/imagenet_select.json"
    elif args.mode_prompt_model_agnostic == "defilip_6_suffix":
        path_template = "data/prompt_template_task_specific/defilip_6_suffix.json"
    elif args.mode_prompt_model_agnostic == "imagenet_80_suffix":
        path_template = "data/prompt_template_task_specific/imagenet_80_suffix.json"
    elif args.mode_prompt_model_agnostic == "imagenet_select_suffix":
        path_template = "data/prompt_template_task_specific/imagenet_select_suffix.json"
    else:
        print("\n\tuse default-imagenet 80 template")
        path_template = "data/prompt_template_task_specific/imagenet_80.json"
    
    with open(path_template, 'r') as f:
        prompt_dict = json.load(f)

    prompt_template_list = []
    if "suffix" not in prompt_dict and "prefix" in prompt_dict:
        for temp_prompt in prompt_dict["prefix"]:
            prompt_template_list.append(temp_prompt.lower())
    elif "suffix" in prompt_dict and "prefix" in prompt_dict:
        prompt_template_list = []
        for temp_suffix in prompt_dict["suffix"]:
            for temp_prefix in prompt_dict["prefix"]: 
                prompt_template_list.append(f"{temp_prefix} {temp_suffix}".lower().strip())

    # ----- add dataset type ----- # 
    if args.add_dataset_species_flag:
        
        print("\t\tadd dataset type")

        # 读取 dataset list # 
        with open(args.path_dataset_species, "r") as f:
            data_dataset_species = json.load(f)

        prompt_template_list = add_dataset_type_to_template(data_dataset_species[args.dataset.lower()], prompt_template_list)

    else:
        dataset_type_dict = {
            "oxford_pets": ", a type of pet.", 
            "flo": ", a type of flower.", 
            "fgvc_aircraft": ", a type of aircraft.", 
            "food101": ", a type of food."
        }

        if args.dataset.lower() in ["oxford_pets", "flo", "fgvc_aircraft", "food101"]:
            prompt_template_new = []
            for temp_prompt in prompt_template_list:
                prompt_template_new.append(temp_prompt.replace(".", dataset_type_dict[args.dataset.lower()]))
            prompt_template_list = prompt_template_new
        elif args.dataset.lower() == "ucf101":
            prompt_template_new = [] 
            for temp_prompt in prompt_template_list:
                prompt_template_new.append(temp_prompt.replace(" {}", " person doing {}"))
            prompt_template_list = prompt_template_new
        elif args.dataset.lower() == "euro_sat":
            prompt_template_new = []
            for temp_prompt in prompt_template_list:
                prompt_template_new.append(temp_prompt.replace("photo", "centered satellite photo"))
            prompt_template_list = prompt_template_new

    # ----- add Baseline prompt ----- # 
    init_prompt = init_dataset_prompt(args, args.dataset.lower())[0]
    prompt_template_list.remove(init_prompt)
    prompt_template_list.insert(0, init_prompt)
    
    return prompt_template_list


def build_task_specific_prompt_memory(args, model, tokenizer, class_name_list, task_specific_prompt_list, train_image_features_array, train_labels_array, threshold_worst_decrease):
    """build class agnostic memory"""

    path_record_memory = os.path.join(args.task_analysis_dir, f"{args.task_name}_seed_{args.seed_data}_thre_{threshold_worst_decrease}_memory.json")
    path_record_memory_feature = os.path.join(args.task_analysis_dir, f"{args.task_name}_seed_{args.seed_data}_thre_{threshold_worst_decrease}_memory_feature.pt")
    path_task_specific_prompt_list = os.path.join(args.task_analysis_dir, f"{args.task_name}_seed_{args.seed_data}_thre_{threshold_worst_decrease}_memory_feature.npy")

    if (os.path.exists(path_record_memory) and os.path.exists(path_record_memory_feature) and os.path.exists(path_task_specific_prompt_list)) and args.memory_init_flag:

        
        with open(path_record_memory, 'r') as f:
            memory_prompt_dict = json.load(f)
        
        memory_prompt_feature = torch.load(path_record_memory_feature)

        task_specific_prompt_list = np.load(path_task_specific_prompt_list)
        task_specific_prompt_list = task_specific_prompt_list.tolist()

    else:

        prompt_dict = get_prompt_dict(args, task_specific_prompt_list, class_name_list)
        
        text_features = get_task_specific_text_features_without_mean(model, tokenizer, prompt_dict, batch_size=args.batch_size_text)
        print("text_features: ", text_features.shape) 

        memory_prompt_dict = {}
        memory_prompt_feature = {}

        print("\t\tIterate through each prompt")
        for idx, temp_prompt_template in enumerate(tqdm(task_specific_prompt_list)):
            if idx == 0:
                
                memory_prompt_dict["baseline"] = {}
                memory_prompt_dict[temp_prompt_template] = {}
                memory_prompt_dict["delete_prompt"] = []

                avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(train_image_features_array, train_labels_array, text_features[:, idx, :], model, return_detail_flag=True, analysis_flag=True)
                
                acc_per_class = extra_info["acc_per_class"]
                Baseline_acc_per_class = acc_per_class.clone().detach()
                
                # memory_prompt_dict["baseline"]["avg acc"] = avg_acc
                # memory_prompt_dict["baseline"]["avg acc-5"] = avg_acc_5
                memory_prompt_dict["baseline"]["top1"] = acc_top1
                memory_prompt_dict["baseline"]["top5"] = acc_top5
                memory_prompt_dict["baseline"]["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
                # memory_prompt_dict["baseline"]["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
                memory_prompt_dict["baseline"]["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
                # memory_prompt_dict["baseline"]["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
                memory_prompt_dict["baseline"]["fitness_score"] = get_fitness_score(args, memory_prompt_dict["baseline"])

                
                # memory_prompt_dict[temp_prompt_template]["avg acc"] = avg_acc
                # memory_prompt_dict[temp_prompt_template]["avg acc-5"] = avg_acc_5
                memory_prompt_dict[temp_prompt_template]["top1"] = acc_top1
                memory_prompt_dict[temp_prompt_template]["top5"] = acc_top5
                memory_prompt_dict[temp_prompt_template]["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
                # memory_prompt_dict[temp_prompt_template]["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
                memory_prompt_dict[temp_prompt_template]["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
                # memory_prompt_dict[temp_prompt_template]["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
                memory_prompt_dict[temp_prompt_template]["fitness_score"] = get_fitness_score(args, memory_prompt_dict[temp_prompt_template])


                memory_prompt_feature[temp_prompt_template] = text_features[:, idx, :].clone().detach()

            else:
                avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(train_image_features_array, train_labels_array, text_features[:, idx, :], model, return_detail_flag=True, analysis_flag=True)
                acc_per_class = extra_info["acc_per_class"]
                
                
                if memory_prompt_dict["baseline"]["top1"] - acc_top1 >= threshold_worst_decrease:
                    # print(memory_prompt_dict["baseline"]["top1"] - acc_top1)
                    memory_prompt_dict["delete_prompt"].append(temp_prompt_template)
                else:
                    memory_prompt_dict[temp_prompt_template] = {}
                    # memory_prompt_dict[temp_prompt_template]["avg acc"] = avg_acc
                    # memory_prompt_dict[temp_prompt_template]["avg acc-5"] = avg_acc_5
                    memory_prompt_dict[temp_prompt_template]["top1"] = acc_top1
                    memory_prompt_dict[temp_prompt_template]["top5"] = acc_top5
                    memory_prompt_dict[temp_prompt_template]["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
                    # memory_prompt_dict[temp_prompt_template]["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
                    memory_prompt_dict[temp_prompt_template]["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
                    # memory_prompt_dict[temp_prompt_template]["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]

                    memory_prompt_dict[temp_prompt_template]["fitness_score"] = get_fitness_score(args, memory_prompt_dict[temp_prompt_template])

                    memory_prompt_feature[temp_prompt_template] = text_features[:, idx, :].clone().detach()

        if len(task_specific_prompt_list) > args.num_max_prompt and args.num_max_prompt != 0:
            
            task_specific_prompt_list = []

            memory_prompt_dict_sorted = {}
            for temp_key in memory_prompt_dict:
                if temp_key in ["baseline", "delete_prompt"]:
                    continue 

                memory_prompt_dict_sorted[temp_key] = memory_prompt_dict[temp_key]

            memory_prompt_dict_sorted = sorted_dict_by_key(memory_prompt_dict_sorted, key="fitness_score")
            for idx, temp_prompt in enumerate(memory_prompt_dict_sorted):
                task_specific_prompt_list.append(temp_prompt)
            
                if idx+1 == args.num_max_prompt:
                    break  
        
        for temp_prompt in memory_prompt_dict["delete_prompt"]:
            if temp_prompt in task_specific_prompt_list:
                task_specific_prompt_list.remove(temp_prompt)

        save_json(memory_prompt_dict, path_record_memory) 
        np.save(path_task_specific_prompt_list, task_specific_prompt_list)
        torch.save(memory_prompt_feature, path_record_memory_feature)

    len_delete = len(memory_prompt_dict["delete_prompt"])
    print(f"\n\tRemove: {len_delete}, Remain: {len(task_specific_prompt_list)}")
        
    return memory_prompt_dict, memory_prompt_feature, task_specific_prompt_list 
    