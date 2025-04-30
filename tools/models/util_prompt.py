# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from models.description.DCLIP_util import make_descriptor_sentence

# ==================== Functions ==================== #
def get_prompt_from_template_and_description(prompt_agnostic, prompt_specific):

    prompt_all_list = []
    for temp_agnostic in prompt_agnostic:
        for temp_specific in prompt_specific:

            if "**+*+**" in temp_specific:
                temp_label = temp_specific.split("**+*+**")[0]
                temp_description = temp_specific.split("**+*+**")[1]
                temp_prompt = temp_agnostic.format(temp_label) + " " + temp_description
            else:
                temp_prompt = temp_agnostic.format(temp_specific)
            
            prompt_all_list.append(temp_prompt.lower())

    return prompt_all_list


def generate_prompt_list_ID(prompt_to_id_dict, prompt_list):
    """get prompt ID"""

    str_init = "0"*len(prompt_to_id_dict)
    str_init = list(str_init)

    for temp_prompt in prompt_list:

        if temp_prompt in prompt_to_id_dict:
            str_init[prompt_to_id_dict[temp_prompt]] = "1"

    str_init = ''.join(str_init)

    return str_init
    

def init_dataset_prompt(args, dataset):

    prompt_template = ["a photo of a {}."]

    if not args.add_dataset_species_flag:
        dataset_type_dict = {
            "oxford_pets": ", a type of pet.", 
            "flo": ", a type of flower.", 
            "fgvc_aircraft": ", a type of aircraft.", 
            "food101": ", a type of food."
        }

        if dataset in ["oxford_pets", "flo", "fgvc_aircraft", "food101"]:
            prompt_template_new = []
            for temp_prompt in prompt_template:
                prompt_template_new.append(temp_prompt.replace(".", dataset_type_dict[dataset]))
            prompt_template = prompt_template_new
        elif dataset == "ucf101":
            prompt_template_new = [] 
            for temp_prompt in prompt_template:
                prompt_template_new.append(temp_prompt.replace(" {}", " person doing {}"))
            prompt_template = prompt_template_new
        elif dataset == "euro_sat":
            prompt_template_new = []
            for temp_prompt in prompt_template:
                prompt_template_new.append(temp_prompt.replace("photo", "centered satellite photo"))
            prompt_template = prompt_template_new

    return prompt_template


def get_prompt_dict(args, prompt_template, class_name_list):
            
    prompt_dict = {}
    for idx, temp_class in enumerate(class_name_list):
        prompt_dict[temp_class] = []
        for temp_prompt in prompt_template:
            prompt_dict[temp_class].append(temp_prompt.format(temp_class))

    return prompt_dict

def add_dataset_type_to_template(dataset_type_list, template_list):
    
    template_list_add_type = []

    for temp_template in template_list:

        template_list_add_type.append(temp_template)

        for temp_type in dataset_type_list:

            temp_type = temp_type.lower()

            template_list_add_type.append(temp_template.replace(".", f", a type of {temp_type}."))

            template_list_add_type.append(temp_template.replace(" {}", ''.join([f" {temp_type}:", " {}"])))

            if "photo" in temp_template:
                template_list_add_type.append(temp_template.replace("photo", temp_type))

                template_list_add_type.append(temp_template.replace("photo", f"{temp_type} photo"))

    return template_list_add_type


def get_full_prompt_from_json_path(class_name_list, path_description):
    
    if not os.path.exists(path_description):
        print(f"{path_description} not exists !")
        sys.exit()

    with open(path_description, 'r') as f:
        data_description = json.load(f)

    prompt_list = []

    for i in range(len(class_name_list)):

        if class_name_list[i].replace("-", " ") in data_description:
            prompt_list.append(data_description[class_name_list[i].replace("-", " ")])
        else:
            prompt_list.append(data_description[class_name_list[i]])

    return prompt_list

def get_prompt_with_description(prompt_list, class_name_list, mode_prompt_class_specific, path_description, dataset_name):
    
    if mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT"]:
        path_description = os.path.join("data/dataset", dataset_name, f"category_description/{mode_prompt_class_specific}.json") 
    elif mode_prompt_class_specific in ["ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
        path_description = os.path.join("data/dataset", dataset_name, f"category_description/{mode_prompt_class_specific}.json") 

    if not os.path.exists(path_description):
        print(f"{path_description} not exists !")
        sys.exit()

    with open(path_description, 'r') as f:
        data_description = json.load(f)

    prompt_list_new = []
    for i in range(len(class_name_list)):
        temp_category = class_name_list[i]
        temp_prompt_list = prompt_list[i]

        temp_prompt_list_new = []
        for temp_prompt in temp_prompt_list:
            for temp_description in data_description[temp_category]:
                if mode_prompt_class_specific == "DCLIP": 
                    descriptor = make_descriptor_sentence(temp_description)
                    temp_prompt_with_description = f"{temp_prompt.replace('.', '')}, {descriptor}."
                else:
                    temp_prompt_with_description = f"{temp_prompt} {temp_description}"

                temp_prompt_list_new.append(temp_prompt_with_description)

        prompt_list_new.append(temp_prompt_list_new)

    return prompt_list_new

def get_prompt(args, prompt_template_list, class_name_list, path_category_synonym=None, add_dataset_species_flag=False, dataset_name=None):
    
    prompt_list = []

    if add_dataset_species_flag:
        with open("data/dataset/dataset_type.json", 'r') as f:
            dataset_type_json = json.load(f)
            dataset_type = dataset_type_json[dataset_name]


    if path_category_synonym is not None:
        with open(path_category_synonym, 'r') as f:
            data_category_synonym = json.load(f)

        for temp_class in class_name_list:
            temp_class_prompt_list = []
            temp_class_synonym = data_category_synonym[temp_class]

            for temp_prompt_template in prompt_template_list:
                for temp_temp_class in temp_class_synonym:
                    if add_dataset_species_flag and dataset_type != "":
                        temp_class_prompt_list.append(temp_prompt_template.format(f"{dataset_type}: a {temp_temp_class}"))
                    else:
                        temp_class_prompt_list.append(temp_prompt_template.format(temp_temp_class))

            prompt_list.append(temp_class_prompt_list)

    else:
        for temp_class in class_name_list:
            temp_class_prompt_list = []
            for temp_prompt_template in prompt_template_list:
                if add_dataset_species_flag:
                    temp_class_prompt_list.append(temp_prompt_template.format(f"{dataset_type}: a {temp_class}"))
                else:
                    temp_class_prompt_list.append(temp_prompt_template.format(temp_class))

            prompt_list.append(temp_class_prompt_list)

    return prompt_list




def load_prompt_classes_agnostic(args, mode, path_document=None):
    
    if mode in ["clip", "class_name", "defilip_6", "filip", "imagenet_80", 
                "imagenet_select", "defilip_6_suffix", "imagenet_80_suffix", "imagenet_select_suffix"]:
        path_document = f"data/prompt_template_task_specific/{mode}.json"

    
    try:
        with open(path_document, 'r') as f:
            prompt_dict = json.load(f) 
    except:
        sys.exit()

    if mode == "json_list":

        prompt_template_json_list = {}
        for temp_name in prompt_dict:

            prompt_template_list = []

            temp_prompt_dict = prompt_dict[temp_name]

            if "suffix" not in temp_prompt_dict and "prefix" in temp_prompt_dict:
                prompt_template_list = temp_prompt_dict["prefix"]
            elif "suffix" in temp_prompt_dict and "prefix" in temp_prompt_dict:
                prompt_template_list = []
                for temp_suffix in temp_prompt_dict["suffix"]:
                    for temp_prefix in temp_prompt_dict["prefix"]: 
                        prompt_template_list.append(f"{temp_prefix} {temp_suffix}".strip())
            else:
                sys.exit()

            prompt_template_json_list[temp_name] = prompt_template_list

        return prompt_template_json_list 
    
    else:
        prompt_template_list = [] 

        if "suffix" not in prompt_dict and "prefix" in prompt_dict:
            prompt_template_list = prompt_dict["prefix"]
        elif "suffix" in prompt_dict and "prefix" in prompt_dict:
            prompt_template_list = []
            for temp_suffix in prompt_dict["suffix"]:
                for temp_prefix in prompt_dict["prefix"]: 
                    prompt_template_list.append(f"{temp_prefix} {temp_suffix}".strip())
        else:
            sys.exit()

    if not args.add_dataset_species_flag and not args.a_photo_of_a_flag:

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

    return prompt_template_list
    
def get_prompt_list_with_args(args, class_name_list):
    if args.full_prompt_flag:
        if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT", "ablation_query_prompt_1", "ablation_query_prompt_2", "ablation_query_prompt_3", "ablation_query_prompt_4", "ablation_description_10", "ablation_description_25", "ablation_description_75", "ablation_description_100"]:
            path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 
        else:
            path_description = args.path_description

        prompt_list = get_full_prompt_from_json_path(class_name_list, path_description)
    else:
        prompt_template_list = load_prompt_classes_agnostic(args, args.mode_prompt_model_agnostic, args.path_prompt_model_agnostic)
        prompt_list = get_prompt(args, prompt_template_list, class_name_list, args.path_prompt_synonym_label, args.add_dataset_species_flag, args.dataset.lower())

        if args.mode_prompt_class_specific is not None:
            prompt_list = get_prompt_with_description(prompt_list, class_name_list, args.mode_prompt_class_specific, args.path_description, args.dataset.lower())

    return prompt_list

def get_init_prompt_dict(args, class_name_list):
    
    init_prompt = init_dataset_prompt(args, args.dataset.lower())[0]

    prompt_dict = {}
    for temp_category in class_name_list:
        prompt_dict[temp_category] = init_prompt.format(temp_category)

    return prompt_dict