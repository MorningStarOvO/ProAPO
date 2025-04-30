# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 
import random
import copy
import math 

import torch  
import clip 

from tqdm import tqdm  

from models.APO_build_knowledge_base.task_specific_knowledge import get_task_specific_prompt_knowledge_base, build_task_specific_prompt_memory
from models.APO_score_func.get_score_based_ACC import get_acc_scores
from models.util_prompt import get_prompt_dict, generate_prompt_list_ID
from models.util_get_features import get_text_features
from utils.util_model import random_weighted_choice, init_record_result, update_best_result, init_weight_dict, get_fitness_score
from utils.util import save_json, sorted_dict, sorted_dict_by_key
from models.util_get_features import get_text_features_from_prompt_list


# ==================== Functions ==================== #
def iterative_select_task_specific_optimal_prompt(args, model, tokenizer, class_name_list, 
                                                   train_image_features_array, train_labels_array, test_image_features_array, test_labels_array, 
                                                   threshold_worst_decrease, exp_name):
    
    if not args.debug:
        if not os.path.exists(args.task_dir):
            data_task_info = {}
            data_task_info = init_record_result(data_task_info, "best model")

            save_json(data_task_info, args.task_dir)

        with open(args.task_dir, 'r') as f:
            data_task_info = json.load(f)

        if not os.path.exists(f"{args.task_prompt_dir}_agnostic.json"):
            best_prompt = {}
            # best_prompt["avg acc"] = {}
            # best_prompt["avg acc-5"] = {}
            best_prompt["top1"] = {}
            best_prompt["top5"] = {}
            
            save_json(best_prompt, f"{args.task_prompt_dir}_agnostic.json")

        with open(f"{args.task_prompt_dir}_agnostic.json", 'r') as f:
            best_prompt = json.load(f) 

        if exp_name in data_task_info:
            if "finish" in data_task_info[exp_name]:
                with open(f"{args.task_prompt_dir}_agnostic.json", 'r') as f:
                    best_prompt = json.load(f)
                return best_prompt        
        
        data_task_info[exp_name] = {}
        data_task_info[exp_name]["top1"] = 0 
        data_task_info[exp_name]["top5"] = 0
        data_task_info[exp_name]["detailed epoch"] = {}
                
        # path_record_prompt = os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}_prompt.json")
        path_record_score = os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}_score.json")
        # path_record_composition = os.path.join(args.task_analysis_dir, f"{args.task_name}_{exp_name}_composition.txt")

    # ---------- step1: Load Library ---------- # 
    print("\t\tload task-specific template library")
    task_specific_prompt_list = get_task_specific_prompt_knowledge_base(args)

    if len(task_specific_prompt_list) == 0:
        sys.exit()

    memory_prompt_dict, memory_prompt_feature, task_specific_prompt_list = build_task_specific_prompt_memory(args, model, tokenizer, class_name_list, task_specific_prompt_list, 
                                                                                                               train_image_features_array, train_labels_array, threshold_worst_decrease)
    
    if args.prompt_sample_mode == "weighted":
        weighted_prompt_dict = {}
        for temp_prompt in task_specific_prompt_list:
            res_score =  memory_prompt_dict[temp_prompt]["fitness_score"] - memory_prompt_dict["baseline"]["fitness_score"]
            if args.weighted_mode == "linear":
                weighted_prompt_dict[temp_prompt] = res_score + args.threshold_epsilon_linear
            elif args.weighted_mode == "exp":
                weighted_prompt_dict[temp_prompt] = np.exp(res_score)

    prompt_to_id_dict = {}
    id_prompt = 0
    for temp_prompt in task_specific_prompt_list:
        if temp_prompt not in prompt_to_id_dict:
            prompt_to_id_dict[temp_prompt] = id_prompt
            id_prompt += 1

    list_record_composition = []


    if args.add_mode == "add_more":
        num_add_max = int(np.log2(len(task_specific_prompt_list))) - int(np.log2(args.epochs)) 

        print("num_add_max: ", num_add_max)

        if num_add_max <= 0:
            num_add_max = 1

        add_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
        remove_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
        replace_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
        mutation_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
        crossover_more_dict = init_weight_dict(num_add_max, args.weighted_mode, args.threshold_epsilon_linear)
        
    # ---------- step2: init ---------- # 
    print("\t\tinit prompt and score =>")
    record_prompt_dict, record_score_dict = init_task_specific_text_prompt_features_and_score(args, memory_prompt_dict, task_specific_prompt_list)
    record_prompt_dict_test = {}
    record_score_dict_test = {}
        
    # ---------- step3: iterative optimization ---------- # 
    model.eval()

    print("\t\titerative optimization =>")
    num_iterative_prompt = args.num_top * (args.num_add + args.num_remove + args.num_replace) + args.num_mutation + args.num_crossover
    print(f"\t\tThe number of iterative Prompts is: {num_iterative_prompt}")

    for epoch in tqdm(range(args.epochs + 1)):
        args.epoch_now = epoch
        
        record_score_dict = sorted_dict_by_key(record_score_dict, key="fitness_score")

        save_json(record_score_dict, path_record_score)
        
        if epoch >= args.epochs:
            break 

        top_key_list = []
        for idx, temp_key in enumerate(record_score_dict):
            top_key_list.append(temp_key)
            if idx == 0:
                top_1_key = temp_key
            if idx + 1 == args.num_top:
                break 
    
        if len(top_key_list) < args.num_top:
            top_key_list = random.choices(top_key_list, k=args.num_top)
        
        # ---------- Generation ----------# 
        for idx_top, temp_top_key in enumerate(top_key_list):

            temp_prompt_template = record_prompt_dict[temp_top_key]
            
            temp_task_specific_prompt_list = task_specific_prompt_list.copy()
            for temp_prompt in temp_prompt_template:
                if temp_prompt in temp_task_specific_prompt_list:
                    temp_task_specific_prompt_list.remove(temp_prompt)

            if len(temp_task_specific_prompt_list) <= 0:
                continue

            if args.add_mode == "add_more":
                temp_num_add_max =  int(np.log2(len(temp_task_specific_prompt_list)))
                if temp_num_add_max > num_add_max:
                    temp_num_add_max = num_add_max

            if args.prompt_sample_mode == "weighted":
                temp_weighted_prompt_list = []
                for temp_prompt in temp_task_specific_prompt_list:
                    temp_weighted_prompt_list.append(weighted_prompt_dict[temp_prompt]) 

            top_prompt_id = generate_prompt_list_ID(prompt_to_id_dict, temp_prompt_template)
            if top_prompt_id not in list_record_composition:
                list_record_composition.append(top_prompt_id)
                          
            # ----- add ----- # 
            num_true_generate = 0
            if args.num_add > 0:
                for index_add in range(args.num_add + 64): 
                    
                    temp_key_str = f"epoch_{epoch}-idx_top_{idx_top}-index_add_{index_add}"

                    if args.add_mode == "add_one":
                        temp_num_add = 1
                    elif args.add_mode == "add_more":
                        if args.prompt_sample_mode == "weighted":
                            num_choice = random.choices(list(add_more_dict.keys()), weights=list(add_more_dict.values()))[0]
                            
                        else:
                            num_choice = random.choices(list(add_more_dict.keys()))[0]
                        temp_num_add = math.pow(2, num_choice)
                        temp_num_add = int(temp_num_add)

                    if temp_num_add > len(temp_task_specific_prompt_list):
                        continue
                        
                    if args.prompt_sample_mode == "weighted":
                        temp_prompt_list = random_weighted_choice(temp_task_specific_prompt_list, temp_weighted_prompt_list, temp_num_add)
                    else:
                        temp_prompt_list = np.random.choice(temp_task_specific_prompt_list, temp_num_add, replace=False)
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

                    record_prompt_dict, record_score_dict = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, temp_add_prompt,
                                                                                            record_prompt_dict, record_score_dict, temp_key_str, memory_prompt_dict, memory_prompt_feature)

                    if args.fitness_analysis_flag:
                        record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, temp_add_prompt,
                                                                                                record_prompt_dict_test, record_score_dict_test, temp_key_str, memory_prompt_dict, memory_prompt_feature, print_flag=True)

                        record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]

                    
                    if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                        if args.weighted_mode == "linear":
                            add_more_dict[num_choice] = add_more_dict[num_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                        elif args.weighted_mode == "exp":
                            add_more_dict[num_choice] = add_more_dict[num_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                        
                    
                    if num_true_generate == args.num_add:
                        break 
            
            if args.add_mode == "add_more":
                num_local_max = int(np.log2(len(temp_prompt_template)))

            if args.prompt_sample_mode == "weighted":
                temp_weighted_temp_prompt_list = []
                for temp_prompt in temp_prompt_template:
                    temp_weighted_temp_prompt_list.append(weighted_prompt_dict[temp_prompt])

            # ----- remove ----- # 
            if len(temp_prompt_template) > 2 and args.num_remove > 0:

                num_true_generate = 0
                
                for index_remove in range(args.num_remove + 64):
                    
                    temp_key_str = f"epoch_{epoch}-idx_top_{idx_top}-index_remove_{index_remove}"
                    
                    if args.add_mode == "add_more": 
                        if args.prompt_sample_mode == "weighted":
                            num_remove_choice = random.choices(list(remove_more_dict.keys())[:num_local_max+1], weights=list(remove_more_dict.values())[:num_local_max+1])[0]
                        else:
                            num_remove_choice = random.choices(list(range(num_local_max+1)))[0]
                        temp_num_remove = math.pow(2, num_remove_choice)
                        temp_num_remove = int(temp_num_remove)
                    else:
                        temp_num_remove = 1

                    if temp_num_remove >= len(temp_prompt_template):
                        continue

                    if args.prompt_sample_mode == "weighted":
                        temp_prompt_list = random_weighted_choice(temp_prompt_template, temp_weighted_temp_prompt_list, temp_num_remove, reverse=True)
                    else:
                        temp_prompt_list = np.random.choice(temp_prompt_template, temp_num_remove, replace=False)
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

                    record_prompt_dict, record_score_dict = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, temp_remove_prompt,
                                                                                              record_prompt_dict, record_score_dict, temp_key_str, memory_prompt_dict, memory_prompt_feature)

                    if args.fitness_analysis_flag:
                        record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, temp_remove_prompt,
                                                                                                record_prompt_dict_test, record_score_dict_test, temp_key_str, memory_prompt_dict, memory_prompt_feature, print_flag=True)
                        record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]

                    if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                        if args.weighted_mode == "linear":
                            remove_more_dict[num_remove_choice] = remove_more_dict[num_remove_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                        elif args.weighted_mode == "exp":
                            remove_more_dict[num_remove_choice] = remove_more_dict[num_remove_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                    
                    if num_true_generate == args.num_remove:
                        break 

            
            # ----- replace ----- # 
            if args.num_replace > 0:
                num_true_generate = 0

                for index_replace in range(args.num_replace + 64):

                    temp_key_str = f"epoch_{epoch}-idx_top_{idx_top}-index_replace_{index_replace}"

                    if args.add_mode == "add_more": 
                        if args.prompt_sample_mode == "weighted":
                    
                            num_replace_choice = random.choices(list(replace_more_dict.keys())[:num_local_max+1], weights=list(replace_more_dict.values())[:num_local_max+1])[0]
                        else:
                            num_replace_choice = random.choices(list(range(num_local_max+1)))[0]
                        temp_num_replace = math.pow(2, num_replace_choice)
                        temp_num_replace = int(temp_num_replace)
                    else:
                        temp_num_replace = 1

                    if temp_num_replace > len(temp_prompt_template):
                        continue

                    if args.prompt_sample_mode == "weighted":
                        temp_select_prompt_list = random_weighted_choice(temp_prompt_template, temp_weighted_temp_prompt_list, temp_num_replace, reverse=True)
                        temp_replace_prompt_list = random_weighted_choice(temp_task_specific_prompt_list, temp_weighted_prompt_list, temp_num_replace)
                    else:
                        temp_select_prompt_list = np.random.choice(temp_prompt_template, temp_num_replace, replace=False)
                        temp_replace_prompt_list = np.random.choice(temp_task_specific_prompt_list, temp_num_replace, replace=False)

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

                    record_prompt_dict, record_score_dict = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, temp_replace_prompt,
                                                                                            record_prompt_dict, record_score_dict, temp_key_str, memory_prompt_dict, memory_prompt_feature)
                    
                    if args.fitness_analysis_flag:
                        record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, temp_replace_prompt,
                                                                                                record_prompt_dict_test, record_score_dict_test, temp_key_str, memory_prompt_dict, memory_prompt_feature, print_flag=True)
                        record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]

                    if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                        if args.weighted_mode == "linear":
                            replace_more_dict[num_replace_choice] = replace_more_dict[num_replace_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                        elif args.weighted_mode == "exp":
                            replace_more_dict[num_replace_choice] = replace_more_dict[num_replace_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                        
                    if num_true_generate == args.num_replace:
                        break 
            
        # ---------- mutation ---------- # 
        if args.num_mutation > 0:
            num_true_generate = 0
            for idx_mutation in range(args.num_mutation + 64):

                temp_key_str = f"epoch_{epoch}-index_mutation_{idx_mutation}"

                if args.add_mode == "add_more": 
                    if args.prompt_sample_mode == "weighted":
                        # print("list(mutation_more_dict.keys())[:num_local_max]: ", list(mutation_more_dict.keys())[:num_local_max])
                        # print("list(mutation_more_dict.values())[:num_local_max]: ", list(mutation_more_dict.values())[:num_local_max])

                        num_mutation_choice = random.choices(list(mutation_more_dict.keys())[:num_local_max+1], weights=list(mutation_more_dict.values())[:num_local_max+1])[0]
                    else:
                        num_mutation_choice = random.choices(list(range(num_local_max+1)))[0]
                    temp_num_mutation = math.pow(2, num_mutation_choice)
                    temp_num_mutation = int(temp_num_mutation) + 2
                else:
                    temp_num_mutation = 2 + epoch

                if temp_num_mutation > len(task_specific_prompt_list):
                    continue 

                if args.prompt_sample_mode == "weighted":
                    temp_mutation_prompt = random_weighted_choice(task_specific_prompt_list, list(weighted_prompt_dict.values()), temp_num_mutation)
                else:
                    temp_mutation_prompt = np.random.choice(task_specific_prompt_list, temp_num_mutation, replace=False)
                    temp_mutation_prompt = temp_mutation_prompt.tolist()

                temp_prompt_mutation_id = generate_prompt_list_ID(prompt_to_id_dict, temp_mutation_prompt)
                if temp_prompt_mutation_id in list_record_composition:
                    continue 
                else:
                    list_record_composition.append(temp_prompt_mutation_id)
                    num_true_generate += 1
                
                record_prompt_dict, record_score_dict = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, temp_mutation_prompt,
                                                                                          record_prompt_dict, record_score_dict, temp_key_str, memory_prompt_dict, memory_prompt_feature)

                if args.fitness_analysis_flag:
                    record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, temp_mutation_prompt,
                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, memory_prompt_dict, memory_prompt_feature, print_flag=True)
                    record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]        


                if args.prompt_sample_mode == "weighted" and args.add_mode == "add_more":
                    if args.weighted_mode == "linear":
                        mutation_more_dict[num_mutation_choice] = mutation_more_dict[num_mutation_choice] * args.weighted_momentum + (record_score_dict[temp_key_str]["fitness_score"] + args.threshold_epsilon_linear - memory_prompt_dict["baseline"]["fitness_score"]) * (1 - args.weighted_momentum)
                    elif args.weighted_mode == "exp":
                        mutation_more_dict[num_mutation_choice] = mutation_more_dict[num_mutation_choice] * args.weighted_momentum + np.exp(record_score_dict[temp_key_str]["fitness_score"] - args.threshold_epsilon_exp - memory_prompt_dict["baseline"]["fitness_score"])  * (1 - args.weighted_momentum)
                        
                if num_true_generate == args.num_mutation:
                    break

        # ---------- crossover ---------- # 
        if args.num_crossover > 0:
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
            for temp_prompt in crossover_prompt_dict:
                if crossover_prompt_dict[temp_prompt] == args.num_top:
                    common_template.append(temp_prompt)
                else:
                    diff_template[temp_prompt] = crossover_prompt_dict[temp_prompt]

            if len(diff_template) > 1:
                for id_inner_crossover in range(args.num_crossover + 64):

                    id_outer_crossover = random.randint(1, len(diff_template))

                    temp_key_str = f"epoch_{epoch}-crossover-inner_{id_inner_crossover}-num_{id_outer_crossover}"

                    temp_crossover_prompt = random_weighted_choice(list(diff_template.keys()), list(diff_template.values()), id_outer_crossover)
                    temp_crossover_prompt.extend(common_template)

                    temp_prompt_crossover_id = generate_prompt_list_ID(prompt_to_id_dict, temp_crossover_prompt)
                    if temp_prompt_crossover_id in list_record_composition:
                        continue 
                    else:
                        list_record_composition.append(temp_prompt_crossover_id)
                        num_true_generate += 1

                    record_prompt_dict, record_score_dict = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, train_image_features_array, train_labels_array, temp_crossover_prompt,
                                                                                                record_prompt_dict, record_score_dict, temp_key_str, memory_prompt_dict, memory_prompt_feature)
                    
                    if args.fitness_analysis_flag:
                        record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, temp_crossover_prompt,
                                                                                                            record_prompt_dict_test, record_score_dict_test, temp_key_str, memory_prompt_dict, memory_prompt_feature, print_flag=True)
                        record_score_dict[temp_key_str]["top1_test"] = record_score_dict_test[temp_key_str]["top1"]

                    if num_true_generate == args.num_crossover:
                        break


        top_key_list = []
        for idx, temp_key in enumerate(record_score_dict):
            top_key_list.append(temp_key)
            if idx == 0:
                top_1_key = temp_key
                break 
        

        record_prompt_dict_test, record_score_dict_test = evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, test_image_features_array, test_labels_array, record_prompt_dict[top_1_key],
                                                                                            record_prompt_dict_test, record_score_dict_test, top_1_key, memory_prompt_dict, memory_prompt_feature, print_flag=True)
            

        data_task_info[exp_name]["detailed epoch"][epoch] = record_score_dict_test[top_1_key]
         
        torch.cuda.empty_cache()

    
    data_task_info[exp_name]["top1"] = record_score_dict_test[top_1_key]["top1"]
    data_task_info[exp_name]["top5"] = record_score_dict_test[top_1_key]["top5"] 

    data_task_info["best model"], best_prompt, update_flag = update_best_result(data_task_info["best model"], record_score_dict_test[top_1_key], exp_name, best_prompt, record_prompt_dict[top_1_key])
    save_json(data_task_info, args.task_dir)

    if update_flag:
        save_json(best_prompt, f"{args.task_prompt_dir}_agnostic.json")
        update_flag = 0


    if args.dataset.lower() != "imagenet":
        data_task_info[exp_name]["finish"] = 1
    save_json(data_task_info, args.task_dir)
    print(f"exp best: ", data_task_info[exp_name]["top1"])
    print("task best: ", data_task_info["best model"])

    return best_prompt


def init_task_specific_text_prompt_features_and_score(args, memory_prompt_dict, task_specific_prompt_list):
    
    record_score_dict = {}
    record_prompt_dict = {}
    
    if args.prompt_initial_mode == "default":

        prompt_template = "a photo of a {}."
        key_init = "default"   
        
        record_score_dict[key_init] = memory_prompt_dict["a photo of a {}."]
        record_prompt_dict[key_init] = [prompt_template]

    elif args.prompt_initial_mode == "sample_manual":

        if args.num_top > len(task_specific_prompt_list):
            sample_prompt_list = task_specific_prompt_list.copy()
        else:
            sample_prompt_list = np.random.choice(task_specific_prompt_list, args.num_top, replace=False)
            sample_prompt_list = sample_prompt_list.tolist()
        
        
        for idx, temp_prompt in enumerate(sample_prompt_list):
            prompt_template = temp_prompt

            if idx == 0:
                key_init = f"default"   
            else:
                key_init = f"epoch0-{idx}" 
            
            record_prompt_dict[key_init] = [prompt_template]
            record_score_dict[key_init] = memory_prompt_dict[prompt_template]
            
    elif args.prompt_initial_mode == "select_top":

        memory_prompt_dict_sorted = {}
        for temp_key in memory_prompt_dict:
            if temp_key in ["baseline", "delete_prompt"]:
                continue 

            memory_prompt_dict_sorted[temp_key] = memory_prompt_dict[temp_key]

        memory_prompt_dict_sorted = sorted_dict_by_key(memory_prompt_dict_sorted, key="fitness_score")
        for idx, temp_prompt in enumerate(memory_prompt_dict_sorted):
            prompt_template = temp_prompt
            key_init = f"epoch0-{idx}" 

            record_prompt_dict[key_init] = [prompt_template]
            record_score_dict[key_init] = memory_prompt_dict[prompt_template]

            if idx >= args.num_top:
                break  

    return record_prompt_dict, record_score_dict


def evaluate_agnostic_prompt_template(args, model, tokenizer, class_name_list, image_features, labels, prompt_template, 
                             record_prompt_dict, record_score_dict, key_str, memory_prompt_dict, memory_prompt_feature, print_flag=False):
    
    if len(prompt_template) == 1 and prompt_template[0] in memory_prompt_feature and print_flag==False:
        record_prompt_dict[key_str] = prompt_template
        record_score_dict[key_str] = memory_prompt_dict[prompt_template[0]]
        return record_prompt_dict, record_score_dict

    with torch.no_grad():
        model.eval()

        appear_flag = 1
        for temp_prompt in prompt_template:
            if temp_prompt not in memory_prompt_feature:
                appear_flag = 0 

        if appear_flag == 1:
            
            for idx, temp_prompt in enumerate(prompt_template):
                if idx == 0:
                    text_features = memory_prompt_feature[temp_prompt].unsqueeze(0)
                else:
                    text_features = torch.cat((text_features, memory_prompt_feature[temp_prompt].unsqueeze(0)))

            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            prompt_dict = get_prompt_dict(args, prompt_template, class_name_list)

            text_features = get_text_features(model, tokenizer, prompt_dict)

        avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(image_features, labels, text_features, model, return_detail_flag=True, analysis_flag=True)

    record_prompt_dict[key_str] = prompt_template
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

    if print_flag:
        print(f"acc_top1: {acc_top1}, acc_top5: {acc_top5}")

    return record_prompt_dict, record_score_dict


def remove_similar_template(task_specific_prompt_list, model, tokenizer, max_similar_score=0.98):
    
    template_text_features = get_text_features_from_prompt_list(model, tokenizer, task_specific_prompt_list)
    similar_score = template_text_features @ template_text_features.t()

    I = torch.eye(similar_score.shape[0]).cuda()
    similar_score = torch.triu(similar_score - I)

    delete_index_list = set((similar_score > max_similar_score).nonzero()[:, 1].tolist())
    
    task_specific_prompt_list_new = []
    for idx, temp_template in enumerate(task_specific_prompt_list):
        if idx not in delete_index_list:
            task_specific_prompt_list_new.append(temp_template)
        
    print(f"\n\tscore: {max_similar_score}, remove: {len(delete_index_list)}, remain: {len(task_specific_prompt_list_new)}")

    return task_specific_prompt_list_new 