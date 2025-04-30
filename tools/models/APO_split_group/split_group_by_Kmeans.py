# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 
import random

from sklearn import metrics
from sklearn.cluster import KMeans

import faiss  
from faiss import normalize_L2

import torch  
import clip 

import heapq

from tqdm import tqdm  

from utils.util import save_json, sorted_dict, sorted_dict_by_key
from models.util_prompt import get_prompt_dict, get_full_prompt_from_json_path
from models.util_get_features import get_text_features
from models.APO_score_func.get_score_based_ACC import get_acc_scores, get_acc_scores_and_dict_quick_and_group, get_acc_scores_based_on_logits
from utils.util_model import get_fitness_score
from models.description.DCLIP_util import make_descriptor_sentence
from models.util_prompt import generate_prompt_list_ID, get_prompt_from_template_and_description

# ==================== Functions ==================== #
def split_group_based_on_Kmeans(args, class_name_list, text_features):
    
    if args.ablation_mode == "ablation_with_all_class_in_one_group":
        group_to_category_dict = {}
        group_to_category_dict[str(0)] = class_name_list

        category_to_group_dict = {}
        for temp_category in class_name_list:
            category_to_group_dict[temp_category] = 0

        return group_to_category_dict, category_to_group_dict

    text_features = text_features.clone().detach().cpu().numpy()

    scores = []
    max_k = 0
    max_score = 0
    max_y_pred = None

    if int(len(class_name_list) * args.num_K_means_max_rate) < 5:
        min_value = 2
        max_value = 3
    else:
        min_value = 5
        max_value = int(len(class_name_list) * args.num_K_means_max_rate)


    for k in tqdm(range(min_value, max_value, 2)):

        dim = text_features.shape[1]
        kmeans = faiss.Kmeans(dim, k, niter=150, seed=args.seed) 
        kmeans.train(text_features)

        D, y_pred = kmeans.index.search(text_features, 1) 
        y_pred = np.squeeze(y_pred)

        score = metrics.silhouette_score(text_features, y_pred)
        scores.append(score)

        if score > max_score:
            max_score = score 
            max_k = k
            max_y_pred = y_pred.copy()

    print("\tThe optimal number of groups is: ", max_k, ", score: ", max_score)

    if max_k == 0:
        max_k = 1

    class_name_list = np.array(class_name_list)
    group_to_category_dict = {}
    category_to_group_dict = {}
    for i in range(max_k):
        group_to_category_dict[str(i)] = list(class_name_list[np.where(max_y_pred == i)])

        for temp_category in group_to_category_dict[str(i)]:
            category_to_group_dict[temp_category] = i 
            
    return group_to_category_dict, category_to_group_dict


def get_category_group_split_based_Kmeans(args, model, tokenizer, class_name_list, image_features, labels, prompt_dict, prompt_template):
    
    text_features = get_text_features(model, tokenizer, prompt_dict) # [category_num, feature_dim]
    
    group_to_category_dict, category_to_group_dict = split_group_based_on_Kmeans(args, class_name_list, text_features)
     

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


    category_group_split_new = {}
    num_consider = 0
    group_id = 0
    delete_category_list = []

    if not args.unable_category_important_score_flag and args.ablation_mode != "ablation_with_all_class_in_one_group":
        print("\nMeasuring the importance of each category ...")
        category_important_score_dict = evaluate_category_important_score(args, model, acc_dict, tokenizer, image_features, text_features, labels, prompt_template, class_name_list, baseline_result)

        for temp_category in category_important_score_dict:
            
            if temp_category == "baseline" or (category_important_score_dict[temp_category] < category_important_score_dict["baseline"] + 0.1 and num_consider > 1):
                break 

            if group_id not in category_group_split_new:
                category_group_split_new[group_id] = []

            if len(category_group_split_new[group_id]) >= args.num_consider_category_each_group:
                group_id += 1
                category_group_split_new[group_id] = []

            category_group_split_new[group_id].append(temp_category)

            num_consider += 1

        if (not args.delete_category_base_important_score_flag) and args.num_bad_delete_rate > 0:
            num_bad_consider = 0 
            max_bad_consider = int (len(class_name_list) * args.num_bad_delete_rate)
            category_important_score_worst_dict = sorted_dict(category_important_score_dict, reverse=False)
            for temp_category in category_important_score_worst_dict:
                if temp_category == "baseline" or num_bad_consider >= max_bad_consider:
                    break 

                delete_category_list.append(temp_category)
                num_bad_consider += 1

        if num_consider > 0:
            group_id += 1
        
    if args.not_consider_performance_flag:

        category_select_dict = {}
        for temp_category in class_name_list:
            category_select_dict[temp_category] = 0 
    
        for temp_group in category_group_split:

            temp_error_category = category_group_split[temp_group][0]

            if category_select_dict[temp_error_category]:
                continue 

            temp_K_means_group = group_to_category_dict[str(category_to_group_dict[temp_error_category])]

            category_group_split_new[group_id + int(temp_group)] = []
            for temp_temp_category in temp_K_means_group:
                if temp_temp_category not in delete_category_list:
                    category_group_split_new[group_id + int(temp_group)].append(temp_temp_category)

            for temp_category in category_group_split[temp_group]:
                category_select_dict[temp_category] = 1

            num_consider += len(temp_K_means_group)
            if num_consider >= args.max_num_consider and args.max_num_consider > 0:
                break 

        category_group_split = category_group_split_new.copy()

    else:

        category_select_dict = {}
        for temp_category in class_name_list:
            category_select_dict[temp_category] = 0 

        for temp_group in category_group_split:
            
            temp_error_category = category_group_split[temp_group][0]
            if category_select_dict[temp_error_category]:
                continue 
                        
            category_group_split_new[group_id + int(temp_group)] = []
            for temp_temp_category in category_group_split[temp_group]:
                if temp_temp_category not in delete_category_list:
                    category_group_split_new[group_id + int(temp_group)].append(temp_temp_category)
            

            for temp_category in category_group_split[temp_group]:
                category_select_dict[temp_category] = 1

            temp_K_means_group = group_to_category_dict[str(category_to_group_dict[temp_error_category])]
            for temp_category in temp_K_means_group:
                if temp_category not in category_group_split_new[group_id + int(temp_group)] and not category_select_dict[temp_category] and temp_category not in delete_category_list:
                    category_group_split_new[group_id + int(temp_group)].append(temp_category)
                    category_select_dict[temp_category] = 1

                    if len(category_group_split_new[group_id + int(temp_group)]) >= args.num_consider_category_each_group:
                        break 

            num_consider += len(category_group_split_new[group_id + int(temp_group)])
            if num_consider >= args.max_num_consider and args.max_num_consider > 0:
                break 

        category_group_split = category_group_split_new.copy()


    print("\nThe total number of current groups is: ", len(category_group_split))

    return category_group_split, text_features, baseline_result


def init_category_group_split_based_Kmeans(args, model, tokenizer, class_name_list, image_features, labels, prompt_template):
    
    if args.description_sample_flag:
        print("\n\tSampling descriptions ...")

        path_prompt_template_key_dict = os.path.join(args.task_analysis_dir, f"{args.seed}_sample_template_key.json")
        path_prompt_dict = os.path.join(args.task_analysis_dir, f"{args.seed}_sample_prompt_dict.json")
        path_text_features = os.path.join(args.task_analysis_dir, f"{args.seed}_sample_text_features.pt")

        if os.path.exists(path_prompt_template_key_dict) and os.path.exists(path_prompt_dict) and os.path.exists(path_text_features):
            
            with open(path_prompt_template_key_dict, 'r') as f:
                prompt_template_key_dict = json.load(f)

            with open(path_prompt_dict, 'r') as f:
                prompt_dict = json.load(f)

            text_features = torch.load(path_text_features)
            text_features = text_features.cuda()
        else:

            if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
                path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 
            else:
                path_description = args.path_description

            with open(path_description, 'r') as f:
                data_description = json.load(f)

            best_fitness_score = -1e6
            best_prompt_template_key_dict = {}
            best_prompt_dict = {}
            best_text_features = 0
            for id_sample in tqdm(range(args.num_description_sample_random + args.num_description_sample_same + 2)):

                if id_sample == 0:
                    prompt_template_key_dict = {}
                    for temp_category in class_name_list:
                        temp_description_list = data_description[temp_category]
                        prompt_template_key_dict[temp_category] = []
                        for temp_description in temp_description_list:
                            if args.mode_prompt_class_specific == "DCLIP":
                                temp_description = make_descriptor_sentence(temp_description)
                            prompt_template_key_dict[temp_category].append(f"{temp_category.lower()}**+*+**{temp_description.lower()}")

                elif id_sample == 1:
                    prompt_template_key_dict = {} 
                    for temp_category in class_name_list:
                        prompt_template_key_dict[temp_category] = [f"{temp_category.lower()}"]

                elif id_sample > 1 and id_sample < args.num_description_sample_random + 2:
                    prompt_template_key_dict = {}
                    for temp_category in class_name_list:
                        prompt_template_key_dict[temp_category] = []

                        temp_num_sample = random.randint(int(len(data_description[temp_category])/3+0.71), len(data_description[temp_category]))
                        if temp_num_sample > len(data_description[temp_category]):
                            temp_num_sample = len(data_description[temp_category])
                        temp_description_list = random.sample(data_description[temp_category], temp_num_sample)
                        for temp_description in temp_description_list:
                            if args.mode_prompt_class_specific == "DCLIP":
                                temp_description = make_descriptor_sentence(temp_description)
                            prompt_template_key_dict[temp_category].append(f"{temp_category.lower()}**+*+**{temp_description.lower()}")
                
                else:
                    prompt_template_key_dict = {}

                    temp_num_sample = random.randint(int(len(data_description[temp_category])/3+0.71), len(data_description[temp_category]))
                    for temp_category in class_name_list:
                        prompt_template_key_dict[temp_category] = []
                        if temp_num_sample > len(data_description[temp_category]):
                            temp_num_sample = len(data_description[temp_category])
                            
                        temp_description_list = random.sample(data_description[temp_category], temp_num_sample)
                        
                        for temp_description in temp_description_list:
                            if args.mode_prompt_class_specific == "DCLIP":
                                temp_description = make_descriptor_sentence(temp_description)
                            prompt_template_key_dict[temp_category].append(f"{temp_category.lower()}**+*+**{temp_description.lower()}")

                prompt_dict = {}
                for temp_category in class_name_list:
                    prompt_dict[temp_category] = get_prompt_from_template_and_description(prompt_template, prompt_template_key_dict[temp_category])

                text_features = get_text_features(model, tokenizer, prompt_dict) 

                avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(image_features, labels, text_features, model, return_detail_flag=True, analysis_flag=True)

                temp_result = {}
                # temp_result["avg acc"] = avg_acc
                # temp_result["avg acc-5"] = avg_acc_5
                temp_result["top1"] = acc_top1
                temp_result["top5"] = acc_top5
                temp_result["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
                # temp_result["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
                temp_result["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
                # temp_result["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]
                temp_result["fitness_score"] = get_fitness_score(args, temp_result)

                print(temp_result["fitness_score"])

                if temp_result["fitness_score"] > best_fitness_score:
                    best_fitness_score = temp_result["fitness_score"]
                    best_prompt_template_key_dict = prompt_template_key_dict.copy()
                    best_prompt_dict = prompt_dict.copy()
                    best_text_features = text_features.clone().detach() 

            # sys.exit()

            prompt_template_key_dict = best_prompt_template_key_dict
            prompt_dict = best_prompt_dict
            text_features = best_text_features

            save_json(prompt_template_key_dict, path_prompt_template_key_dict)
            save_json(prompt_dict, path_prompt_dict)
            torch.save(text_features.clone().detach().cpu(), path_text_features)

    elif args.description_init_flag:
        print("\n\t读取 description ing ...")

        if args.mode_prompt_class_specific not in [None, "definition"]:
            path_description_init = os.path.join(args.path_dataset_output, "features", f"{args.mode_prompt_class_specific}_{args.path_best_agnostic_prompt}.json")
            path_prompt_template_key_dict = os.path.join(args.path_dataset_output, "features", f"{args.mode_prompt_class_specific}_{args.path_best_agnostic_prompt}_template_key_dict.json")
            path_text_features_init = os.path.join(args.path_dataset_output, "features", f"{args.mode_prompt_class_specific}_{args.path_best_agnostic_prompt}.pt")
            cache_flag = 1
        else:
            cache_flag = 0
            path_description_init = None
            path_text_features_init = None
            path_prompt_template_key_dict = None 

        if cache_flag and os.path.exists(path_description_init) and os.path.exists(path_text_features_init) and os.path.exists(path_prompt_template_key_dict):
            with open(path_description_init, 'r') as f:
                prompt_dict = json.load(f)

            with open(path_prompt_template_key_dict, 'r') as f:
                prompt_template_key_dict = json.load(f)

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
                save_json(prompt_template_key_dict, path_prompt_template_key_dict)
                torch.save(text_features.clone().detach().cpu(), path_text_features_init)
            
        
    else:
        prompt_dict = get_prompt_dict(args, prompt_template, class_name_list)

        text_features = get_text_features(model, tokenizer, prompt_dict) # [category_num, feature_dim]\

        if not args.unable_same_template_flag:
            prompt_template_key_dict = {}
            for temp_category in class_name_list:
                prompt_template_key_dict[temp_category] = [temp_category.lower()]

    # print("text_features: ", text_features.shape)

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

    # print("baseline_result: ", baseline_result)
    # sys.exit()

    path_group_init = os.path.join(args.task_analysis_dir, f"{args.task_name}-seed_{args.seed}-group_init.json") 
    if os.path.exists(path_group_init):

        with open(path_group_init, 'r') as f:
            category_group_split = json.load(f)        

        print("\n category_group_split: ", category_group_split)
        print("\nThe total number of current groups is: ", len(category_group_split))

        # sys.exit()

        if not args.unable_same_template_flag:
            return category_group_split, prompt_dict, prompt_template_key_dict, text_features, baseline_result
        else:
            return category_group_split, prompt_dict, text_features, baseline_result
    else:
    
        group_to_category_dict, category_to_group_dict = split_group_based_on_Kmeans(args, class_name_list, text_features)

        category_group_split_new = {}
        num_consider = 0
        group_id = 0
        delete_category_list = []

        if not args.unable_category_important_score_flag:
            print("\nMeasuring the importance of each category ...")
            category_important_score_dict = evaluate_category_important_score(args, model, acc_dict, tokenizer, image_features, text_features, labels, prompt_template, class_name_list, baseline_result)

            for temp_category in category_important_score_dict:
                
                if temp_category == "baseline" or (category_important_score_dict[temp_category] < category_important_score_dict["baseline"] + 0.1 and num_consider > 1):
                    break 

                if group_id not in category_group_split_new:
                    category_group_split_new[group_id] = []

                if len(category_group_split_new[group_id]) >= args.num_consider_category_each_group:
                    group_id += 1
                    category_group_split_new[group_id] = []

                category_group_split_new[group_id].append(temp_category)

                num_consider += 1

            if (not args.delete_category_base_important_score_flag) and args.num_bad_delete_rate > 0:
                num_bad_consider = 0 
                max_bad_consider = int (len(class_name_list) * args.num_bad_delete_rate)
                category_important_score_worst_dict = sorted_dict(category_important_score_dict, reverse=False)
                for temp_category in category_important_score_worst_dict:
                    if temp_category == "baseline" or num_bad_consider >= max_bad_consider:
                        break 

                    delete_category_list.append(temp_category)
                    num_bad_consider += 1

                print("delete_category_list: ", delete_category_list)

            if num_consider > 0:
                group_id += 1
            
        if args.not_consider_performance_flag:

            category_select_dict = {}
            for temp_category in class_name_list:
                category_select_dict[temp_category] = 0 

            
            for temp_group in category_group_split:

                temp_error_category = category_group_split[temp_group][0]
            
                if category_select_dict[temp_error_category]:
                    continue 

                temp_K_means_group = group_to_category_dict[str(category_to_group_dict[temp_error_category])]

                category_group_split_new[group_id + int(temp_group)] = []
                for temp_temp_category in temp_K_means_group:
                    if temp_temp_category not in delete_category_list:
                        category_group_split_new[group_id + int(temp_group)].append(temp_temp_category)

                for temp_category in category_group_split[temp_group]:
                    category_select_dict[temp_category] = 1

            
                num_consider += len(temp_K_means_group)
                if num_consider >= args.max_num_consider and args.max_num_consider > 0:
                    break 

            category_group_split = category_group_split_new.copy()

        else:

            category_select_dict = {}
            for temp_category in class_name_list:
                category_select_dict[temp_category] = 0 

            for temp_group in category_group_split:
                
                temp_error_category = category_group_split[temp_group][0]
                if category_select_dict[temp_error_category]:
                    continue 
                
                category_group_split_new[group_id + int(temp_group)] = []
                for temp_temp_category in category_group_split[temp_group]:
                    if temp_temp_category not in delete_category_list:
                        category_group_split_new[group_id + int(temp_group)].append(temp_temp_category)
                
                for temp_category in category_group_split[temp_group]:
                    category_select_dict[temp_category] = 1

                temp_K_means_group = group_to_category_dict[str(category_to_group_dict[temp_error_category])]
                for temp_category in temp_K_means_group:
                    if temp_category not in category_group_split_new[group_id + int(temp_group)] and not category_select_dict[temp_category] and temp_category not in delete_category_list:
                        category_group_split_new[group_id + int(temp_group)].append(temp_category)
                        category_select_dict[temp_category] = 1

                        if len(category_group_split_new[group_id + int(temp_group)]) >= args.num_consider_category_each_group:
                            break 

                num_consider += len(category_group_split_new[group_id + int(temp_group)])
                if num_consider >= args.max_num_consider and args.max_num_consider > 0:
                    break 

            category_group_split = category_group_split_new.copy()

        print("\nThe total number of current groups is: ", len(category_group_split), ", Number of categories considered: ", num_consider)
        # print("category_group_split: ", category_group_split)


        save_json(category_group_split, path_group_init)
        print("category_group_split: ", category_group_split)

        if not args.unable_same_template_flag:
            return category_group_split, prompt_dict, prompt_template_key_dict, text_features, baseline_result
        else:
            return category_group_split, prompt_dict, text_features, baseline_result


def evaluate_category_important_score(args, model, acc_dict, tokenizer, image_features, text_features, labels, prompt_template, class_name_list, baseline_result):
    
    description_flag = 0
    if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
        path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 

        if not os.path.exists(path_description):
            print(f"{path_description} not exists !")
            sys.exit()

        with open(path_description, 'r') as f:
            data_description = json.load(f)

        description_flag = 1

    if not args.unable_synonym_label_flag:
        path_label_synonym = os.path.join("data/dataset", args.dataset.lower(), "category_synonym", f"{args.category_synonym_mode}.json")
        with open(path_label_synonym, 'r') as f:
            data_label_synonym = json.load(f)

    category_to_label_dict = {}
    for idx, temp_category in enumerate(class_name_list):
        category_to_label_dict[temp_category] = idx 

    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

    category_important_score_dict = {}
    category_important_score_dict["baseline"] = baseline_result["fitness_score"]
    # for temp_category in tqdm(class_name_list):
    for i, temp_category in tqdm(enumerate(acc_dict)):

        
        if args.unable_synonym_label_flag:
            temp_label_synonym = [temp_category]
        else:
            temp_label_synonym = data_label_synonym[temp_category]

        temp_category_prompt_list = []

        for temp_synonym in temp_label_synonym:
            if temp_synonym not in temp_category_prompt_list:
                temp_category_prompt_list.append(temp_synonym)

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
                    temp_combine_prompt = f"{temp_synonym}*****{temp_description_new}"

                    if temp_combine_prompt not in temp_category_prompt_list:
                        temp_category_prompt_list.append(temp_combine_prompt)

        with torch.no_grad():
            model.eval()
            text_features_modified = text_features.clone().detach()      
            logits_modified = logits.clone().detach() 

            temp_class_prompts = get_prompt_from_template_and_description(prompt_template, temp_category_prompt_list)
            temp_prompts = tokenizer(temp_class_prompts).cuda()

            temp_text_features = model.encode_text(temp_prompts)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            temp_text_features = temp_text_features.mean(dim=0)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)
            text_features_modified[category_to_label_dict[temp_category], :] = temp_text_features

            logit_scale = model.logit_scale.exp()
            temp_logits = logit_scale * image_features @ temp_text_features.unsqueeze(0).t()
            logits_modified[:, category_to_label_dict[temp_category]] = temp_logits[:, 0]

            # avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores(image_features, labels, text_features_modified, model, return_detail_flag=True, analysis_flag=True)
            avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info = get_acc_scores_based_on_logits(text_features_modified, logits_modified, labels, model, return_detail_flag=True, analysis_flag=True)
            
            temp_result = {}
            
            # temp_result["avg acc"] = avg_acc
            # temp_result["avg acc-5"] = avg_acc_5
            temp_result["top1"] = acc_top1
            temp_result["top5"] = acc_top5
            temp_result["redundant_score_l2_norm"] = extra_info["redundant_score_l2_norm"]
            # temp_result["redundant_score_most_similar"] = extra_info["redundant_score_most_similar"]
            temp_result["inter_class_score_CE"] = extra_info["inter_class_score_CE"]
            # temp_result["inter_class_score_most_similar"] = extra_info["inter_class_score_most_similar"]

            category_important_score_dict[temp_category] = get_fitness_score(args, temp_result)

            torch.cuda.empty_cache()


    category_important_score_dict = sorted_dict(category_important_score_dict)
    
    return category_important_score_dict
