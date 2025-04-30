# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 
from copy import deepcopy

import torch  

from data_process.named_data import get_named_data_ZSL as get_named_data
from models.util_get_features import get_image_features, get_text_features
from models.APO_score_func.get_score_based_ACC import get_acc_scores
from models.util_prompt import get_prompt_dict
from utils.util import save_json


# ==================== Functions ==================== #
def test_prompt_OOD_performance(args, best_prompt, model, tokenizer, preprocess, exp_name, class_name_list):
    
    if not args.debug:
        with open(args.task_dir, 'r') as f:
            data_task_info = json.load(f)

        if "imagenet_a" not in data_task_info["best model"]:
            for temp_dataset in ["imagenet_a", "imagenet_r", "imagenet_s", "imagenet_v2"]:
                data_task_info["best model"][temp_dataset] = 0 

        for temp_dataset in ["imagenet_a", "imagenet_r", "imagenet_s", "imagenet_v2"]:
            data_task_info[exp_name][temp_dataset] = 0
        
    original_data_root = deepcopy(args.data_root)
    dataset_to_root_dict = {
        "imagenet_a": "ImageNet-A/imagenet-a", 
        "imagenet_r": "ImageNet-R/imagenet-r", 
        "imagenet_s": "ImageNet-S/sketch", 
        "imagenet_v2": "ImageNet-V2/imagenetv2-matched-frequency-format-val"
    }
    for temp_dataset in ["imagenet_a", "imagenet_r", "imagenet_s", "imagenet_v2"]:

        args.dataset = temp_dataset
        args.data_root = original_data_root.replace("ImageNet/val", dataset_to_root_dict[temp_dataset])
        test_loader, class_name_list, image_file_dict = get_named_data(args, preprocess)

        if temp_dataset in ["imagenet_a", "imagenet_r"]:
            with open(os.path.join("data/dataset", temp_dataset, "label_to_category_name.json"), 'r') as f:
                label_to_category_name = json.load(f)
            prompt_dict = {}
            for temp_label in label_to_category_name:
                prompt_dict[label_to_category_name[temp_label]] = best_prompt[label_to_category_name[temp_label]]
        else:
            prompt_dict = best_prompt.copy()
            
        model_name_str = args.model.replace("/", "_").replace("@", "_")
        test_labels, test_image_features = get_image_features(args, test_loader, model, mode=f"{model_name_str}_{temp_dataset}_test")

        with torch.no_grad():
            text_features = get_text_features(model, tokenizer, prompt_dict)
            avg_acc, avg_acc_5, acc_top1, acc_top5 = get_acc_scores(test_image_features, test_labels, text_features, model)
            data_task_info[exp_name][temp_dataset] = acc_top1

    data_task_info[exp_name]["finish"] = 1
    
    for temp_dataset in ["imagenet_a", "imagenet_r", "imagenet_s", "imagenet_v2"]:
        if data_task_info[exp_name][temp_dataset] > data_task_info["best model"][temp_dataset]:
            data_task_info["best model"][temp_dataset] = data_task_info[exp_name][temp_dataset]
            print(f"\nupdate best result - {temp_dataset}: ", data_task_info["best model"][temp_dataset])

        print(f"{temp_dataset}: ", data_task_info[exp_name][temp_dataset])
    
    save_json(data_task_info, args.task_dir)