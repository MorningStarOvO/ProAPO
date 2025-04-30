# ==================== 导入必要的包 ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch  
import clip 

from tqdm import tqdm  

import numpy as np 
import json 
import random
import copy
import math 

from models.APO_build_knowledge_base.class_specific_knowledge import get_class_specific_prompt_to_category, build_class_specific_prompt_memory
from models.APO_score_func.get_score_based_ACC import get_acc_scores
from models.APO_split_group.split_group_based_ACC import init_category_group_split, get_category_group_split
from models.APO_split_group.split_group_by_Kmeans import get_category_group_split_based_Kmeans, init_category_group_split_based_Kmeans
from models.util_get_features import get_text_features, get_text_features_from_prompt_list
from utils.util_model import random_weighted_choice, init_record_result, update_best_result, init_weight_dict, get_fitness_score
from utils.util import save_json, sorted_dict_by_key
from models.util_prompt import generate_prompt_list_ID


# ==================== Functions ==================== #
def iterative_select_class_specific_optimal_prompt(args, model, tokenizer, best_prompt_agnostic, class_name_list, train_image_features_array, train_labels_array, test_image_features_array, test_labels_array, threshold_worst_decrease, exp_name):
    
    # ---------- step0 ---------- # 
    if not args.debug:
        if not os.path.exists(args.task_dir):
            data_task_info = {}
            data_task_info = init_record_result(data_task_info, "best model")

            save_json(data_task_info, args.task_dir)

        with open(args.task_dir, 'r') as f:
            data_task_info = json.load(f)

        if not os.path.exists(f"{args.task_prompt_dir}_dependent.json"):
            best_prompt = {}
            # best_prompt["avg acc"] = {}
            # best_prompt["avg acc-5"] = {}
            best_prompt["top1"] = {}
            best_prompt["top5"] = {}
            
            save_json(best_prompt, f"{args.task_prompt_dir}_dependent.json")

        with open(f"{args.task_prompt_dir}_dependent.json", 'r') as f:
            best_prompt = json.load(f) 

        if exp_name in data_task_info and "finish" in data_task_info[exp_name]:
            return         
        
        data_task_info = init_record_result(data_task_info, exp_name)
        data_task_info[exp_name]["detailed epoch"] = {}
                

    category_to_label_dict = {}
    category_select_dict = {}
    for idx, temp_category in enumerate(class_name_list):
        category_to_label_dict[temp_category] = idx 
        category_select_dict[temp_category] = 0

    best_model_prompt = {}
    best_model_prompt["top1"] = {}
    best_model_prompt["top5"] = {}

    prompt_delete_all_list = []


    # ---------- step1:  ---------- # 
    for epoch_outer in range(args.epochs_outer):
        print(f"\t\tEpoch outer: {epoch_outer} ...")
        args.epoch_outer = epoch_outer

        # ---------- step2: group ---------- #  
        print("\t\t\tgrouping")
        if epoch_outer == 0:
            if args.only_performance_init_flag:
                category_group_split, prompt_dict, text_features, baseline_result = init_category_group_split(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, best_prompt_agnostic)
            else:
                category_group_split, prompt_dict, text_features, baseline_result = init_category_group_split_based_Kmeans(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, best_prompt_agnostic)
        else:
            if args.only_performance_init_flag:
                category_group_split, text_features, baseline_result = get_category_group_split(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, prompt_dict)
            else:
                category_group_split, text_features, baseline_result = get_category_group_split_based_Kmeans(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, prompt_dict)
        
        if args.threshold_group_mode == "under_baseline":
            args.threshold_group_acc = baseline_result["top1"]

        # ---------- step3: Iterate through each group ---------- # 
        for temp_group_id in category_group_split:
            args.temp_group_id = temp_group_id

            if int(temp_group_id) >= args.num_max_search_group:
                break 

            print(f"\t\t\t group-{temp_group_id} ing")

            if epoch_outer == 0 and int(temp_group_id) == 0:
                init_flag = 1

            if init_flag:
                prompt_to_category_dict = {}
            
            temp_group = category_group_split[temp_group_id]
            print("temp_group: ", temp_group)
            if len(temp_group) <= 1:
                continue
                
            # ---------- step4: load Library ---------- # 
            print("\t\t\t\tLoad Library")
            prompt_to_category_dict = get_class_specific_prompt_to_category(args, prompt_dict, temp_group, category_select_dict, prompt_to_category_dict)

            # print("prompt_to_category_dict: ", prompt_to_category_dict.keys())

            prompt_delete_all_list = get_remove_similar_description(list(prompt_to_category_dict.keys()), model, tokenizer, prompt_delete_all_list, max_similar_score=args.delete_max_similar_score)

            
            memory_prompt_dict, class_specific_prompt_list, prompt_to_id_dict = build_class_specific_prompt_memory(args, model, tokenizer, temp_group, category_to_label_dict, prompt_to_category_dict, baseline_result,
                                                                                                                     prompt_dict, text_features, train_image_features_array, train_labels_array, 
                                                                                                                     threshold_worst_decrease, prompt_delete_all_list)
            
            save_json(memory_prompt_dict, os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}_-epoch_outer_{epoch_outer}-group_id_{temp_group_id}-memory.json"))

            
            if len(class_specific_prompt_list) == 0 and not init_flag:
                continue 

            if args.prompt_sample_mode == "weighted":
                weighted_prompt_dict = {}
                for temp_prompt in class_specific_prompt_list:
                    res_score =  memory_prompt_dict[temp_prompt]["fitness_score"] - memory_prompt_dict["baseline"]["fitness_score"]
                    if args.weighted_mode == "linear":
                        weighted_prompt_dict[temp_prompt] = res_score + args.threshold_epsilon_linear
                    elif args.weighted_mode == "exp":
                        weighted_prompt_dict[temp_prompt] = np.exp(res_score)

                        
            baseline_prompt = []
            id_baseline = len(prompt_to_id_dict)
            for temp_category in temp_group:
                temp_category_prompt_list = prompt_dict[temp_category]
                baseline_prompt.extend(temp_category_prompt_list)

                for temp_baseline_prompt in temp_category_prompt_list:
                    if temp_baseline_prompt not in prompt_to_id_dict:
                        prompt_to_id_dict[temp_baseline_prompt] = id_baseline
                        id_baseline += 1

            list_record_composition = []
                                                
            if args.add_mode == "add_more":
                num_add_max = round(np.log2(len(class_specific_prompt_list)) + 0.5) - int(np.log2(args.epochs_inner))
            
                if num_add_max <= 0:
                    num_add_max = 1

                add_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
                remove_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
                replace_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
                mutation_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
                crossover_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
                    
            # ---------- step5: init ---------- # 
            if init_flag:
                print("\ninit prompt and score =>")
                record_prompt_dict, record_score_dict, category_select_dict = init_text_prompt_features_and_score_in_independent(args, memory_prompt_dict, class_specific_prompt_list, prompt_to_category_dict, category_select_dict, 
                                                                                                                                 prompt_dict, temp_group, baseline_prompt)

                init_flag = 0
            else:
                for temp_category in temp_group:
                    if category_select_dict[temp_category] == 0:
                        category_select_dict[temp_category] = 1

                for temp_key_record in record_prompt_dict:
                    for temp_prompt in baseline_prompt:
                        if temp_prompt not in record_prompt_dict[temp_key_record]:
                            record_prompt_dict[temp_key_record].append(temp_prompt)
    
            record_prompt_dict_test = {}
            record_score_dict_test = {}

            # ---------- step6: iterative optimization ---------- # 
            model.eval()

            print("\n iterative optimization=>")
            num_iterative_prompt = args.num_dependent_top * (args.num_dependent_add + args.num_dependent_remove + args.num_dependent_replace) + args.num_dependent_mutation + args.num_dependent_crossover
            print(f"\tNumber of Prompt : {num_iterative_prompt}")
    
            for epoch in tqdm(range(args.epochs_inner+1)):
                args.epoch_now = epoch
                
                record_score_dict = sorted_dict_by_key(record_score_dict, key="fitness_score")

                save_json(record_score_dict, os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}-epoch_outer_{epoch_outer}-group_id_{temp_group_id}_score.json"))
                
                # ---------- Select top prompts ---------- #
                top_key_list = []
                for temp_idx, temp_key in enumerate(record_score_dict):
                    top_key_list.append(temp_key)
                    if temp_idx == 0:
                        top_1_key = temp_key
                    if temp_idx + 1 == args.num_dependent_top:
                        break 

                if len(top_key_list) < args.num_dependent_top:
                    top_key_list = random.choices(top_key_list, k=args.num_dependent_top)

                if args.evaluate_mode == "top5_mean":
                    sys.exit()
                elif args.evaluate_mode == "top":
                    record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, record_prompt_dict[top_1_key], prompt_to_category_dict,
                                                                                                text_features, test_image_features_array, test_labels_array, 
                                                                                                record_prompt_dict_test, record_score_dict_test, top_1_key, print_flag=True)
                data_task_info[exp_name]["detailed epoch"][f"Epoch-outer_{epoch_outer}-group_{temp_group_id}-inter_{epoch}"] = record_score_dict_test[top_1_key]
                data_task_info[exp_name], best_model_prompt, _ = update_best_result(data_task_info[exp_name], record_score_dict_test[top_1_key], top_1_key, best_model_prompt, record_prompt_dict[top_1_key])
                print(f"\ntop1:", record_score_dict_test[top_1_key]["top1"], "top5:", record_score_dict_test[top_1_key]["top5"])
                print(f"exp best: ", data_task_info[exp_name]["top1"])
                data_task_info["best model"], best_prompt, update_flag = update_best_result(data_task_info["best model"], record_score_dict_test[top_1_key], exp_name, best_prompt, record_prompt_dict[top_1_key])
                
                save_json(data_task_info, args.task_dir)

                if update_flag:
                    prompt_dict_best = prompt_dict.copy()

                    for temp_best_prompt in best_prompt["top1"]:
                        temp_update_category = prompt_to_category_dict[temp_best_prompt]
                        prompt_dict_best[temp_update_category].append(temp_best_prompt)

                    save_json(prompt_dict_best, f"{args.task_prompt_dir}_dependent.json")
                    update_flag = 0

                if epoch >= args.epochs_inner or epoch >= len(class_specific_prompt_list) + 1:
                    break 

                # ---------- Generation: ----------# 
                for idx_top, temp_top_key in enumerate(top_key_list):

                    temp_prompt_template = record_prompt_dict[temp_top_key]

                    temp_class_specific_prompt_list = class_specific_prompt_list.copy()
                    for temp_prompt in temp_prompt_template:
                        if temp_prompt in temp_class_specific_prompt_list:
                            temp_class_specific_prompt_list.remove(temp_prompt)

                    if len(temp_class_specific_prompt_list) > 0:
                        if args.prompt_sample_mode == "weighted":
                            temp_weighted_prompt_list = []
                            for temp_prompt in temp_class_specific_prompt_list:
                                temp_weighted_prompt_list.append(weighted_prompt_dict[temp_prompt]) 

                    top_prompt_id = generate_prompt_list_ID(prompt_to_id_dict, temp_prompt_template)
                    if top_prompt_id not in list_record_composition:
                        list_record_composition.append(top_prompt_id)

                    # ----- add ----- # 
                    if args.num_dependent_add > 0 and len(temp_class_specific_prompt_list) > 0:
                        num_true_generate = 0
                        for index_add in range(args.num_dependent_add + 64): 
                            
                            temp_key_str = f"outer_{epoch_outer}-group_{temp_group_id}-epoch_{epoch}-idx_top_{idx_top}-index_add_{index_add}"

                            if args.add_mode == "add_one":
                                temp_num_add = 1
                            elif args.add_mode == "add_more":
                                if args.prompt_sample_mode == "weighted":
                                    num_choice = random.choices(list(add_more_dict.keys()), weights=list(add_more_dict.values()))[0]
                                    
                                else:
                                    num_choice = random.choices(list(add_more_dict.keys()))[0]
                                temp_num_add = math.pow(2, num_choice)
                                temp_num_add = int(temp_num_add)

                            if temp_num_add > len(temp_class_specific_prompt_list):
                                continue
                                
                            if args.prompt_sample_mode == "weighted":
                                temp_prompt_list = random_weighted_choice(temp_class_specific_prompt_list, temp_weighted_prompt_list, temp_num_add)
                            else:
                                temp_prompt_list = np.random.choice(temp_class_specific_prompt_list, temp_num_add, replace=False)
                                temp_prompt_list = temp_prompt_list.tolist()

                            temp_add_prompt = temp_prompt_template.copy()
                            temp_prompt_add_id = list(str(top_prompt_id))
                            for temp_prompt in temp_prompt_list:
                                if temp_prompt not in temp_add_prompt:
                                    temp_add_prompt.append(temp_prompt)
                                    temp_prompt_add_id[prompt_to_id_dict[temp_prompt]] = "1"

                            temp_prompt_add_id = ''.join(temp_prompt_add_id)
                            if temp_prompt_add_id in list_record_composition:
                                continue 
                            else:
                                list_record_composition.append(temp_prompt_add_id)
                                num_true_generate += 1

                            record_prompt_dict, record_score_dict = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_add_prompt, prompt_to_category_dict,
                                                                                              text_features, train_image_features_array, train_labels_array, 
                                                                                              record_prompt_dict, record_score_dict, temp_key_str, print_flag=False)


                            if args.fitness_analysis_flag: 
                                record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_add_prompt, prompt_to_category_dict,
                                                                                                            text_features, test_image_features_array, test_labels_array, 
                                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, print_flag=True)

                                record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]
                            
                            if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                                if args.weighted_mode == "linear":
                                    add_more_dict[num_choice] = add_more_dict[num_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                                elif args.weighted_mode == "exp":
                                    add_more_dict[num_choice] = add_more_dict[num_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                                

                            if num_true_generate == args.num_dependent_add:
                                break 
                    

                    temp_prompt_template_in_group = []
                    for temp_prompt in temp_prompt_template:
                        if prompt_to_category_dict[temp_prompt] in temp_group:
                            temp_prompt_template_in_group.append(temp_prompt)

                    
                    if args.add_mode == "add_more":
                        num_local_max = int(np.log2(len(temp_prompt_template_in_group)))
                        # print("num_local_max: ", num_local_max)

                    if args.prompt_sample_mode == "weighted":
                        temp_weighted_temp_prompt_list = []
                        for temp_prompt in temp_prompt_template_in_group:
                            if temp_prompt in temp_weighted_temp_prompt_list:
                                temp_weighted_temp_prompt_list.append(weighted_prompt_dict[temp_prompt])
                            else:
                                if args.weighted_mode == "linear":
                                    temp_weighted_temp_prompt_list.append(args.threshold_epsilon_linear)
                                elif args.weighted_mode == "exp":   
                                    temp_weighted_temp_prompt_list.append(1)

                    # ----- remove ----- # 
                    if len(temp_prompt_template_in_group) > 2 and args.num_dependent_remove > 0:

                        num_true_generate = 0
                        
                        for index_remove in range(args.num_dependent_remove + 64):
                            
                            temp_key_str = f"outer_{epoch_outer}-group_{temp_group_id}-epoch_{epoch}-idx_top_{idx_top}-index_remove_{index_remove}"
                            
                            if args.add_mode == "add_more": 
                                if args.prompt_sample_mode == "weighted":
                                    num_remove_choice = random.choices(list(remove_more_dict.keys())[:num_local_max+1], weights=list(remove_more_dict.values())[:num_local_max+1])[0]
                                else:
                                    num_remove_choice = random.choices(list(range(num_local_max+1)))[0]
                                temp_num_remove = math.pow(2, num_remove_choice)
                                temp_num_remove = int(temp_num_remove)
                            else:
                                temp_num_remove = 1

                            if temp_num_remove >= len(temp_prompt_template_in_group):
                                continue

                            if args.prompt_sample_mode == "weighted":
                                temp_prompt_list = random_weighted_choice(temp_prompt_template_in_group, temp_weighted_temp_prompt_list, temp_num_remove, reverse=True)
                            else:
                                temp_prompt_list = np.random.choice(temp_prompt_template_in_group, temp_num_remove, replace=False)
                                temp_prompt_list = temp_prompt_list.tolist()

                            temp_remove_prompt = temp_prompt_template.copy()
                            temp_prompt_remove_id = list(str(top_prompt_id))
                            for temp_prompt in temp_prompt_list:
                                temp_remove_prompt.remove(temp_prompt)
                                temp_prompt_remove_id[prompt_to_id_dict[temp_prompt]] = "0"

                            temp_prompt_remove_id = ''.join(temp_prompt_remove_id)
                            if temp_prompt_remove_id in list_record_composition:
                                continue 
                            else:
                                list_record_composition.append(temp_prompt_remove_id)
                                num_true_generate += 1

                            record_prompt_dict, record_score_dict = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_remove_prompt, prompt_to_category_dict,
                                                                                              text_features, train_image_features_array, train_labels_array, 
                                                                                              record_prompt_dict, record_score_dict, temp_key_str,  print_flag=False)

                            if args.fitness_analysis_flag: 
                                record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_remove_prompt, prompt_to_category_dict,
                                                                                                            text_features, test_image_features_array, test_labels_array, 
                                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, print_flag=True)

                                record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]

                            if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                                if args.weighted_mode == "linear":
                                    remove_more_dict[num_remove_choice] = remove_more_dict[num_remove_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                                elif args.weighted_mode == "exp":
                                    remove_more_dict[num_remove_choice] = remove_more_dict[num_remove_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                            
                            if num_true_generate == args.num_dependent_remove:
                                break 

                    # ----- replace ----- # 
                    if args.num_dependent_replace > 0 and len(temp_class_specific_prompt_list) > 0:
                        num_true_generate = 0

                        for index_replace in range(args.num_dependent_replace + 64):

                            temp_key_str = f"outer_{epoch_outer}-group_{temp_group_id}-epoch_{epoch}-idx_top_{idx_top}-index_replace_{index_replace}"

                            if args.add_mode == "add_more": 
                                if args.prompt_sample_mode == "weighted":
                                    num_replace_choice = random.choices(list(replace_more_dict.keys())[:num_local_max+1], weights=list(replace_more_dict.values())[:num_local_max+1])[0]
                                else:
                                    num_replace_choice = random.choices(list(range(num_local_max+1)))[0]
                                temp_num_replace = math.pow(2, num_replace_choice)
                                temp_num_replace = int(temp_num_replace)
                            else:
                                temp_num_replace = 1

                            if temp_num_replace > len(temp_prompt_template_in_group):
                                continue

                            if args.prompt_sample_mode == "weighted":
                                temp_select_prompt_list = random_weighted_choice(temp_prompt_template_in_group, temp_weighted_temp_prompt_list, temp_num_replace, reverse=True)
                                temp_replace_prompt_list = random_weighted_choice(temp_class_specific_prompt_list, temp_weighted_prompt_list, temp_num_replace)
                            else:
                                temp_select_prompt_list = np.random.choice(temp_prompt_template_in_group, temp_num_replace, replace=False)
                                temp_replace_prompt_list = np.random.choice(temp_class_specific_prompt_list, temp_num_replace, replace=False)

                            temp_replace_prompt = temp_prompt_template.copy()
                            for temp_idx_replace in range(temp_num_replace):
                                temp_replace_prompt.remove(temp_select_prompt_list[temp_idx_replace])
                                temp_replace_prompt.append(temp_replace_prompt_list[temp_idx_replace])

                            temp_prompt_replace_id = generate_prompt_list_ID(prompt_to_id_dict, temp_replace_prompt)
                            if temp_prompt_replace_id in list_record_composition:
                                continue 
                            else:
                                list_record_composition.append(temp_prompt_replace_id)
                                num_true_generate += 1

                            record_prompt_dict, record_score_dict = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_replace_prompt, prompt_to_category_dict,
                                                                                              text_features, train_image_features_array, train_labels_array, 
                                                                                              record_prompt_dict, record_score_dict, temp_key_str,  print_flag=False)
                            
                            if args.fitness_analysis_flag: 
                                record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_replace_prompt, prompt_to_category_dict,
                                                                                                            text_features, test_image_features_array, test_labels_array, 
                                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, print_flag=True)

                                record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]


                            if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                                if args.weighted_mode == "linear":
                                    replace_more_dict[num_replace_choice] = replace_more_dict[num_replace_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                                elif args.weighted_mode == "exp":
                                    replace_more_dict[num_replace_choice] = replace_more_dict[num_replace_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                                
                            if num_true_generate == args.num_dependent_replace:
                                break 
                    
                # ---------- mutation ---------- # 
                if args.num_dependent_mutation > 0:
                    num_true_generate = 0
                    for idx_mutation in range(args.num_dependent_mutation + 64):

                        temp_key_str = f"outer_{epoch_outer}-group_{temp_group_id}-epoch_{epoch}-index_mutation_{idx_mutation}"

                        if args.add_mode == "add_more": 
                            if args.prompt_sample_mode == "weighted":
                                num_mutation_choice = random.choices(list(mutation_more_dict.keys())[:num_local_max+1], weights=list(mutation_more_dict.values())[:num_local_max+1])[0]
                            else:
                                num_mutation_choice = random.choices(list(range(num_local_max+1)))[0]
                            temp_num_mutation = math.pow(2, num_mutation_choice)
                            temp_num_mutation = int(temp_num_mutation) + 2
                        else:
                            temp_num_mutation = 2 + epoch

                        if temp_num_mutation > len(class_specific_prompt_list):
                            continue 

                        if args.prompt_sample_mode == "weighted":
                            temp_mutation_prompt = random_weighted_choice(class_specific_prompt_list, list(weighted_prompt_dict.values()), temp_num_mutation)
                        else:
                            temp_mutation_prompt = np.random.choice(class_specific_prompt_list, temp_num_mutation, replace=False)
                            temp_mutation_prompt = temp_mutation_prompt.tolist()

                        temp_mutation_prompt.extend(baseline_prompt)

                        temp_prompt_mutation_id = generate_prompt_list_ID(prompt_to_id_dict, temp_mutation_prompt)
                        if temp_prompt_mutation_id in list_record_composition:
                            continue 
                        else:
                            list_record_composition.append(temp_prompt_mutation_id)
                            num_true_generate += 1
                        
                        record_prompt_dict, record_score_dict = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_mutation_prompt, prompt_to_category_dict,
                                                                                                    text_features, train_image_features_array, train_labels_array, 
                                                                                                    record_prompt_dict, record_score_dict, temp_key_str,  print_flag=False)

                        if args.fitness_analysis_flag: 
                            record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_mutation_prompt, prompt_to_category_dict,
                                                                                                        text_features, test_image_features_array, test_labels_array, 
                                                                                                        record_prompt_dict_test, record_score_dict_test, temp_key_str, print_flag=True)

                            record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]


                        if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                            if args.weighted_mode == "linear":
                                mutation_more_dict[num_mutation_choice] = mutation_more_dict[num_mutation_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                            elif args.weighted_mode == "exp":
                                mutation_more_dict[num_mutation_choice] = mutation_more_dict[num_mutation_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                                
                        if num_true_generate == args.num_dependent_mutation:
                            break

                    
                # ---------- crossover ---------- # 
                if args.num_dependent_crossover > 0:
                    num_true_generate = 0

                    if args.crossover_mode == "quick":
                        top_key_list = []
                        for idx, temp_key in enumerate(record_score_dict):
                            top_key_list.append(temp_key)
                            if idx + 1 == args.num_top:
                                break 

                    crossover_prompt_dict = {}
                    for idx_top, temp_top_key in enumerate(top_key_list):
                        temp_prompt_template = record_prompt_dict[temp_top_key]

                        for temp_prompt in temp_prompt_template:
                            if temp_prompt not in crossover_prompt_dict:
                                crossover_prompt_dict[temp_prompt] = 0
                            crossover_prompt_dict[temp_prompt] += 1

                            
                    common_template = []
                    diff_template = {}
                    num_other_group = 0
                    for temp_prompt in crossover_prompt_dict:
                        if crossover_prompt_dict[temp_prompt] == args.num_top:
                            common_template.append(temp_prompt)
                        else:
                            diff_template[temp_prompt] = crossover_prompt_dict[temp_prompt]

                            if prompt_to_category_dict[temp_prompt] not in temp_group:
                                num_other_group += 1

                    
                    if len(diff_template) > 0:

                        for id_inner_crossover in range(args.num_dependent_crossover + 64):

                            id_outer_crossover = random.randint(num_other_group, len(diff_template))

                            temp_key_str = f"outer_{epoch_outer}-group_{temp_group_id}-epoch_{epoch}-crossover-inner_{id_inner_crossover}-num_{id_outer_crossover}"

                    
                            temp_crossover_prompt = random_weighted_choice(list(diff_template.keys()), list(diff_template.values()), id_outer_crossover)
                            temp_crossover_prompt.extend(common_template)

                            temp_prompt_crossover_id = generate_prompt_list_ID(prompt_to_id_dict, temp_crossover_prompt)
                            if temp_prompt_crossover_id in list_record_composition:
                                continue 
                            else:
                                list_record_composition.append(temp_prompt_crossover_id)
                                num_true_generate += 1

                            record_prompt_dict, record_score_dict = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_crossover_prompt, prompt_to_category_dict,
                                                                                                text_features, train_image_features_array, train_labels_array, 
                                                                                                record_prompt_dict, record_score_dict, temp_key_str,  print_flag=False)
                            
                            if args.fitness_analysis_flag: 
                                record_prompt_dict_test, record_score_dict_test = evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, temp_crossover_prompt, prompt_to_category_dict,
                                                                                                            text_features, test_image_features_array, test_labels_array, 
                                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, print_flag=True)

                                record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]


                            if num_true_generate == args.num_dependent_crossover:
                                break
                    
                    
            torch.cuda.empty_cache()

            save_json(record_score_dict, os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}-epoch_outer_{epoch_outer}-group_id_{temp_group_id}_score.json"))
            # save_json(record_prompt_dict, os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}-epoch_outer_{epoch_outer}-group_id_{temp_group_id}_prompt.json"))

            # ---------- step8: update prompt dict ---------- # 
            prompt_dict, text_features, baseline_result, record_prompt_dict, record_score_dict = update_record_in_group_end(args, model, tokenizer, category_to_label_dict, record_prompt_dict, record_score_dict, prompt_to_category_dict,
                                                                                                                            prompt_dict, text_features, train_image_features_array, train_labels_array)


            


    if args.dataset.lower() != "imagenet":
        data_task_info[exp_name]["finish"] = 1
    save_json(data_task_info, args.task_dir)

    print(f"\nexp best: ", data_task_info[exp_name]["top1"])
    print("\ntask best: ", data_task_info["best model"])

    return best_model_prompt


def update_record_in_group_end(args, model, tokenizer, category_to_label_dict, record_prompt_dict, record_score_dict, prompt_to_category_dict,
                               prompt_dict, text_features, image_features, labels):
    
    record_prompt_dict_new = {}
    record_score_dict_new = {}

    prompt_frequency_dict = {}

    for idx, temp_key in enumerate(record_score_dict):

        temp_key_prompt_list = record_prompt_dict[temp_key]

        for temp_prompt in temp_key_prompt_list:

            if temp_prompt not in prompt_frequency_dict:
                prompt_frequency_dict[temp_prompt] = 0

            prompt_frequency_dict[temp_prompt] += 1

        if idx + 1 == args.num_top:
            break 

    prompt_add_list = []
    for temp_prompt in prompt_frequency_dict:
        if prompt_frequency_dict[temp_prompt] == args.num_top:
            prompt_add_list.append(temp_prompt)


    prompt_category_add_dict = {}
    for temp_prompt in prompt_add_list:
        temp_category = prompt_to_category_dict[temp_prompt]

        if temp_category not in prompt_category_add_dict:
            prompt_category_add_dict[temp_category] = []

        prompt_category_add_dict[temp_category].append(temp_prompt)

    prompt_dict_new = prompt_dict.copy()
    for temp_category in prompt_category_add_dict:
        prompt_dict_new[temp_category] = prompt_category_add_dict[temp_category]

    for idx, temp_key in enumerate(record_score_dict):

        temp_new_key = f"0-outer_{args.epoch_outer}-group_{args.temp_group_id}-init_{idx}"

        record_prompt_dict_new[temp_new_key] = []
        record_score_dict_new[temp_new_key] = record_score_dict[temp_key]
        
        temp_key_prompt_list = record_prompt_dict[temp_key]
        for temp_prompt in temp_key_prompt_list:
            if temp_prompt not in prompt_add_list:
                record_prompt_dict_new[temp_new_key].append(temp_prompt)

                temp_prompt_category = prompt_to_category_dict[temp_prompt]
                if temp_prompt_category in prompt_category_add_dict:
                    for temp_base_prompt in prompt_category_add_dict[temp_prompt_category]:

                        if temp_base_prompt in record_prompt_dict_new[temp_new_key]:
                            break 
                        else:
                            record_prompt_dict_new[temp_new_key].append(temp_base_prompt)

        if idx + 1 == args.num_top:
            break 

    text_features_modified = text_features.clone().detach()
    with torch.no_grad():
        model.eval()
        for temp_category in prompt_category_add_dict:
            temp_class_prompts = prompt_category_add_dict[temp_category]
            temp_prompts = tokenizer(temp_class_prompts).cuda()

            temp_text_features = model.encode_text(temp_prompts)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            temp_text_features = temp_text_features.mean(dim=0)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)
            text_features_modified[category_to_label_dict[temp_category], :] = temp_text_features

        avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(image_features, labels, text_features_modified, model, return_detail_flag=True, analysis_flag=True)


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

    return prompt_dict_new, text_features_modified, baseline_result, record_prompt_dict_new, record_score_dict_new


def evaluate_dependent_prompt(args, model, tokenizer, category_to_label_dict, prompt_add_list, prompt_to_category_dict,
                              text_features, image_features, labels,
                              record_prompt_dict, record_score_dict, key_str, 
                              print_flag=False):

    prompt_category_add_dict = {}
    for temp_prompt in prompt_add_list:

        if temp_prompt.lower() in prompt_to_category_dict:
            temp_category = prompt_to_category_dict[temp_prompt.lower()]
        else:
            temp_category = prompt_to_category_dict[temp_prompt]

        if temp_category not in prompt_category_add_dict:
            prompt_category_add_dict[temp_category] = []

        prompt_category_add_dict[temp_category].append(temp_prompt)


    text_features_modified = text_features.clone().detach()
    with torch.no_grad():
        model.eval()
        for temp_category in prompt_category_add_dict:
            temp_class_prompts = prompt_category_add_dict[temp_category]
            temp_prompts = tokenizer(temp_class_prompts).cuda()

            temp_text_features = model.encode_text(temp_prompts)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            temp_text_features = temp_text_features.mean(dim=0)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)
            text_features_modified[category_to_label_dict[temp_category], :] = temp_text_features

        avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(image_features, labels, text_features_modified, model, return_detail_flag=True, analysis_flag=True)

    record_prompt_dict[key_str] = prompt_add_list
    record_score_dict[key_str] = {}
    # record_score_dict[key_str]["avg acc"] = avg_acc
    # record_score_dict[key_str]["avg acc-5"] = avg_acc_5
    record_score_dict[key_str]["top1"] = acc_top1
    record_score_dict[key_str]["top5"] = acc_top5
    record_score_dict[key_str]["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
    # record_score_dict[key_str]["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
    record_score_dict[key_str]["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
    # record_score_dict[key_str]["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
    record_score_dict[key_str]["fitness_score"] = get_fitness_score(args, record_score_dict[key_str])


    if print_flag and not args.fitness_analysis_flag:
        print(f"acc_top1: {acc_top1}, acc_top5: {acc_top5}")

    return record_prompt_dict, record_score_dict


def init_text_prompt_features_and_score_in_independent(args, memory_prompt_dict, init_prompt_list, prompt_to_category_dict, 
                                                       category_select_dict, prompt_dict, group_list, baseline_prompt):

    record_score_dict = {}
    record_prompt_dict = {}

    record_score_dict["baseline"] = memory_prompt_dict["baseline"]
    record_prompt_dict["baseline"] = baseline_prompt.copy()

    if args.prompt_initial_mode == "default":

        for idx in range(args.num_top):
            key_init = f"default-{idx}"    
            record_score_dict[key_init] = memory_prompt_dict[init_prompt_list[idx]]

            record_prompt_dict[key_init] = baseline_prompt.copy()
            if init_prompt_list[idx] not in baseline_prompt:
                record_prompt_dict[key_init].append(init_prompt_list[idx])
            
            temp_category = prompt_to_category_dict[init_prompt_list[idx]]
            if category_select_dict[temp_category] == 0:
                category_select_dict[temp_category] = 1

    elif args.prompt_initial_mode == "sample_manual":

        if args.num_top > len(init_prompt_list):
            sample_prompt_list = init_prompt_list.copy()
        else:
            sample_prompt_list = random.sample(init_prompt_list, args.num_top)

        for idx, temp_prompt in enumerate(sample_prompt_list):
            key_init = f"epoch0-{idx}" 
            record_score_dict[key_init] = memory_prompt_dict[temp_prompt]

            record_prompt_dict[key_init] = baseline_prompt.copy()
            if temp_prompt not in record_prompt_dict[key_init]:
                record_prompt_dict[key_init].append(temp_prompt)
            
            temp_category = prompt_to_category_dict[temp_prompt]
            if category_select_dict[temp_category] == 0:
                category_select_dict[temp_category] = 1
            
    elif args.prompt_initial_mode == "select_top":

        memory_prompt_dict_sorted = {}
        for temp_key in memory_prompt_dict:
            if temp_key in ["baseline", "delete_prompt"]:
                continue 

            memory_prompt_dict_sorted[temp_key] = memory_prompt_dict[temp_key]

        memory_prompt_dict_sorted = sorted_dict_by_key(memory_prompt_dict_sorted, key="fitness_score")
        for idx, temp_prompt in enumerate(memory_prompt_dict_sorted):
            key_init = f"epoch0-{idx}" 

            record_score_dict[key_init] = memory_prompt_dict[temp_prompt]

            record_prompt_dict[key_init] = baseline_prompt.copy()
            if temp_prompt not in record_prompt_dict[key_init]:
                record_prompt_dict[key_init].append(temp_prompt)

            temp_category = prompt_to_category_dict[temp_prompt]
            if category_select_dict[temp_category] == 0:
                category_select_dict[temp_category] = 1

            if idx >= args.num_top:
                break  

    return record_prompt_dict, record_score_dict, category_select_dict
    

def get_remove_similar_description(prompt_list, model, tokenizer, prompt_delete_all_list, max_similar_score=0.98):
    
    prompt_list_new = []
    for temp_prompt in prompt_list:
        if temp_prompt not in prompt_delete_all_list:
            prompt_list_new.append(temp_prompt)


    template_text_features = get_text_features_from_prompt_list(model, tokenizer, prompt_list_new)
    similar_score = template_text_features @ template_text_features.t()

    I = torch.eye(similar_score.shape[0]).cuda()
    similar_score = torch.triu(similar_score - I)

    delete_index_list = set((similar_score > max_similar_score).nonzero()[:, 1].tolist())
    
    for idx, temp_template in enumerate(prompt_list_new):
        if idx in delete_index_list:
            prompt_delete_all_list.append(temp_template)

    print(f"\n\tnumber of remove description: {len(delete_index_list)} ^.^ ~~")

    return  prompt_delete_all_list