# ==================== Import Packages ==================== #

import time
import sys
import os 

import numpy as np 
import json 

import torch 
import clip 

from einops import rearrange, repeat

from tqdm import tqdm  

from models.util_prompt import load_prompt_classes_agnostic, get_prompt, get_prompt_with_description, get_full_prompt_from_json_path


# ==================== Functions ==================== #
def split_number(total, size):
    
    num_parts = total // size 

    last_part = total % size 

    split_number_list = []
    for i in range(num_parts):
        split_number_list.append((i + 1) * size)

    if last_part != 0:
        split_number_list.append(total)

    return split_number_list

def get_task_specific_text_features_without_mean(model, tokenizer, prompt_dict, batch_size=2048):
    
    with torch.no_grad():

        model.eval()
        
        print("\t\tget Text Features")

        index_num_template_list = []
        prompt_list_all = []
        for idx, temp_class in enumerate(prompt_dict):
            num_template = len(prompt_dict[temp_class])

            if idx == 0: 
                index_num_template_list.append(num_template)
            else:
                index_num_template_list.append(num_template + index_num_template_list[idx-1])

            prompt_list_all.extend(prompt_dict[temp_class])
                
        
        split_number_list = split_number(len(prompt_list_all), batch_size)

        temp_last_number = 0
        print("\t\ttokenizing ...")
        prompts_all = tokenizer(prompt_list_all).cuda()
        for idx, temp_number in enumerate(tqdm(split_number_list)):
            
            temp_text_features = model.encode_text(prompts_all[temp_last_number:temp_number])
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            if idx == 0:
                text_features = temp_text_features
            else:
                text_features = torch.cat((text_features, temp_text_features), dim=0)

            del temp_text_features
            torch.cuda.empty_cache()

            temp_last_number = temp_number
        
        text_features = rearrange(text_features, '(a b) c -> a b c', b=index_num_template_list[0])

    return text_features

def get_text_features_from_prompt_list(model, tokenizer, prompt_list, batch_size=2048):
    
    with torch.no_grad():

        model.eval()
            
        split_number_list = split_number(len(prompt_list), batch_size)

        temp_last_number = 0
        prompts_all = tokenizer(prompt_list).cuda()
        for idx, temp_number in enumerate(split_number_list):
            
            temp_text_features = model.encode_text(prompts_all[temp_last_number:temp_number])
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            if idx == 0:
                text_features = temp_text_features
            else:
                text_features = torch.cat((text_features, temp_text_features), dim=0)

            del temp_text_features
            torch.cuda.empty_cache()

            temp_last_number = temp_number
            
        return text_features
    
def get_text_features(model, tokenizer, prompt_dict):
    
    with torch.no_grad():

        model.eval()
        
        for idx, temp_class in enumerate(prompt_dict):
            temp_class_prompts = prompt_dict[temp_class]
            temp_prompts = tokenizer(temp_class_prompts).cuda()

            temp_text_features = model.encode_text(temp_prompts)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)

            temp_text_features = temp_text_features.mean(dim=0)
            temp_text_features /= temp_text_features.norm(dim=-1, keepdim=True)
            temp_text_features = temp_text_features.unsqueeze(0)

            if idx == 0:
                text_features = temp_text_features
            else:
                text_features = torch.cat((text_features, temp_text_features), dim=0)

            del temp_prompts, temp_text_features
            torch.cuda.empty_cache()

    return text_features

def get_image_features(args, loader, model, mode):
    
    if not os.path.exists(os.path.join(args.path_dataset_output, "features")):
        os.makedirs(os.path.join(args.path_dataset_output, "features"), exist_ok=True)

    path_save_label = os.path.join(args.path_dataset_output, "features", f"label_{mode}.pt") 
    path_save_image_features = os.path.join(args.path_dataset_output, "features", f"image_{mode}.pt")

    if os.path.exists(path_save_label) and os.path.exists(path_save_image_features) and args.memory_image_init_flag:

        labels_array = torch.load(path_save_label)
        image_features_array = torch.load(path_save_image_features)
    else:

        model.eval()

        labels_array = 0 
        image_features_array = 0

        with torch.no_grad():
            for i, (img, label) in enumerate(tqdm(loader)):
                
                img, label = img.cuda(), label.cuda() 

                image_features = model.encode_image(img)
                
                if i == 0:
                    labels_array = label.clone().detach().cpu()
                    image_features_array = image_features.clone().detach().cpu()
                else:
                    labels_array = torch.cat((labels_array, label.clone().detach().cpu()))
                    image_features_array = torch.cat((image_features_array, image_features.clone().detach().cpu()))

            image_features_array = image_features_array / image_features_array.norm(dim=-1, keepdim=True)

        torch.save(labels_array, path_save_label)
        torch.save(image_features_array, path_save_image_features)

    return labels_array.cuda(), image_features_array.cuda()

