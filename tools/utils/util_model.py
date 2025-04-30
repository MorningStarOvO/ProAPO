# ==================== Import Packages ==================== #
import os 

import numpy as np 
import json 
import shutil
import random
import math 

import torch

from pprint import pprint


# ==================== Functions ==================== #
def get_fitness_score(args, score_dict):
    
    return score_dict["top1"] * args.score_top_1_rate \
           + score_dict["top5"] * args.score_top_5_rate \
           + score_dict["redundant_score_l2_norm"] * args.score_redundant_score_l2_norm \
           - score_dict["inter_class_score_CE"] * args.score_inter_class_score_CE \
           
        #    + score_dict["redundant_score_most_similar"] * args.score_redundant_score_most_similar \
        #    - score_dict["inter_class_score_most_similar"] * args.score_inter_class_score_most_similar
           
           
def init_weight_dict(num_max, weighted_mode, threshold_epsilon_linear=1):
    
    weight_dict = {}

    for i in range(num_max):
        if weighted_mode == "linear":
            weight_dict[i] = threshold_epsilon_linear
        elif weighted_mode == "exp":
            weight_dict[i] = 1

    return weight_dict


def random_weighted_choice(data_list, p, num, reverse=False):

    p = np.array(p) + 1e-4
    if (p < 0).any():
        # print(p)
        p = p - p.min() + 1e-4

    # print(p)
    p = p / p.sum()
        
    if reverse:
        p = 1 - p + 1e-4
        p = p / p.sum()

    choice_list = np.random.choice(data_list, num, p=p, replace=False)

    choice_list = choice_list.tolist()
        
    return choice_list

def init_record_result(record_dict, key, value=0):
    record_dict[key] = {}

    record_dict[key]["top1"] = value
    record_dict[key]["top5"] = value

    record_dict[key]["key - top1"] = ""
    record_dict[key]["key - top5"] = ""

    return record_dict


def update_best_result(record_dict, result_dict, result_key, best_prompt=None, result_prompt=None):

    update_flag = 0

    if best_prompt is not None:
        best_prompt_flag = 1
    else:
        best_prompt_flag = 0

    if result_dict["top1"] > record_dict["top1"]:
        record_dict["top1"] = result_dict["top1"]
        record_dict["key - top1"] = result_key
        if best_prompt_flag:
            best_prompt["top1"] = result_prompt
            update_flag = 1

    if result_dict["top5"] > record_dict["top5"]:
        record_dict["top5"] = result_dict["top5"] 
        record_dict["key - top5"] = result_key
        if best_prompt_flag:
            best_prompt["top5"] = result_prompt

    if best_prompt_flag:
        return record_dict, best_prompt, update_flag
    else:
        return record_dict


def setup_seed(seed):
    
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res 
