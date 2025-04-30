# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from torch.utils.data import DataLoader
from torchvision import transforms 
import clip 

from utils.util import save_json
from data_process.dataset.dataset_ZSL import dataset_ZSL


# ==================== Functions ==================== #
def get_named_data_adaptive_prompt_search(args, preprocess):

    if args.dataset.lower() in ["cub", "fgvc_aircraft"] and "clip" in args.model:

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
    
        if args.model in ["clip_ViT-L/14@336px"]:
            transform_val = transforms.Compose([
                                transforms.Resize(384),
                                transforms.CenterCrop(336),
                                transforms.ToTensor(),
                                normalize,
                    ])
        else:
            transform_val = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                    ])
    else:
        transform_val = preprocess
    
    print(f"\tLoad dataset: {args.dataset}")

    with open(os.path.join("data/dataset", args.dataset.lower(), "split.json"), 'r') as f:
        data_dict = json.load(f)
    with open(os.path.join("data/dataset", args.dataset.lower(), "label_to_category_name.json"), 'r') as f:
        label_to_category_name = json.load(f)
    
    if args.validate_set_mode == "fewshot":
        path_train_split = os.path.join("data/dataset", args.dataset.lower(), "fewshot_dataset", f"seed_{args.seed_data}_num_shots_{args.num_shots}.json")

        if args.dataset.lower() == "imagenet" or args.dataset.lower() == "sun397" or args.dataset.lower() == "places365":
            train_data_root = os.path.join(args.data_root.replace(args.data_root.split("/")[-1], "train_fewshot"))
        else:
            train_data_root = args.data_root

    with open(path_train_split, 'r') as f:
        data_train = json.load(f)

    train_set = dataset_ZSL(train_data_root, data_train["train"], label_to_category_name, transform_val)
    test_set = dataset_ZSL(args.data_root, data_dict["test"], label_to_category_name, transform_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.workers, pin_memory=False)

    class_name_list = []
    for temp_label in label_to_category_name:
        class_name_list.append(label_to_category_name[temp_label])

    return train_loader, test_loader, class_name_list


def get_named_data_ZSL(args, preprocess):

    if args.dataset.lower() in ["cub", "fgvc_aircraft"] and "clip" in args.model:

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
    
        if args.model in ["clip_ViT-L/14@336px"]:
            transform_val = transforms.Compose([
                                transforms.Resize(384),
                                transforms.CenterCrop(336),
                                transforms.ToTensor(),
                                normalize,
                    ])
        else:
            transform_val = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                    ])
    else:
        transform_val = preprocess
    
    print(f"\tLoad dataset: {args.dataset}")

    with open(os.path.join("data/dataset", args.dataset.lower(), "split.json"), 'r') as f:
        data_dict = json.load(f)
    with open(os.path.join("data/dataset", args.dataset.lower(), "label_to_category_name.json"), 'r') as f:
        label_to_category_name = json.load(f)
        
    test_set = dataset_ZSL(args.data_root, data_dict["test"], label_to_category_name, transform_val)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.workers, pin_memory=False)
    
    class_name_list = []
    for temp_label in label_to_category_name:
        class_name_list.append(label_to_category_name[temp_label])

    image_file_dict = data_dict["test"]

    return test_loader, class_name_list, image_file_dict
