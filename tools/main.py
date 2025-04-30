# ==================== Import Packages ==================== #
import time
import sys
import os 

sys.path.append("tools")

import numpy as np 
import json 

import torch
import models.clip_modified.clip as clip 
from models.clip_modified.clip import tokenize

from pprint import pprint 

from tqdm import tqdm

from configs.option import parse_opt 
from utils.util import build_save_file, Logger
from utils.util_model import setup_seed 
from data_process.named_data import get_named_data_adaptive_prompt_search as get_named_data
from process.select_optimal_prompt import iterative_select_optimal_prompt
from process.test_clip_zero_shot import test_clip_zero_shot
from process.test_OOD_performance import test_prompt_OOD_performance

import warnings
warnings.filterwarnings('ignore')  


# ==================== main ==================== #
if __name__ == '__main__':
    
    # ---------- step0 ---------- # 
    args = parse_opt() 

    if not args.debug: 
        path_task_save = os.path.join(args.checkpoint_path, args.dataset, "task")
        task_dir = os.path.join(path_task_save, f"{args.task_name}.json")
        args.task_dir = task_dir

        finish_flag = 0
        if os.path.exists(args.task_dir):
            with open(args.task_dir, 'r') as f:
                data_task_json = json.load(f)

            if args.APO_mode == "agnostic":
                if args.model_name in data_task_json:
                    if "finish" in data_task_json[args.model_name]:
                        finish_flag = 1
            elif args.APO_mode in ["dependent", "agnostic_and_dependent"]:
                exp_name_new = f"{args.model_name}_{args.APO_mode}"
                if exp_name_new in data_task_json:
                    if "finish" in data_task_json[exp_name_new]:
                        finish_flag = 1
                    
        if finish_flag:
            print("finished")
            sys.exit()

    if args.seed == "0":
        args.seed = np.random.randint(10000)
    setup_seed(args.seed)
    
    build_save_file(args)

    if args.mode_prompt_class_specific in ["DCLIP", "GPT4Vis", "CuPL_base", "CuPL_full", "AdaptCLIP", "AWT"]:   
        path_description = os.path.join("data/dataset", args.dataset.lower(), f"category_description/{args.mode_prompt_class_specific}.json") 
        if not os.path.exists(path_description):
            print(f"{path_description} not exists !")
            sys.exit()
    
    print("\tseed:", args.seed)
    
    if not args.debug:        
        log_dir = os.path.join(args.path_log_save, f"{args.model_name}.log")

        if os.path.exists(log_dir):
            os.remove(log_dir)
        sys.stdout = Logger(filename=log_dir, stream=sys.stdout)

    # ---------- step2: load model ---------- # 
    print("\nload model =>")
    model_type = args.model.split("_")[0]
    model_name = args.model.split("_")[1]


    if model_type == "clip":
        model, preprocess = clip.load(model_name, device="cuda")
        tokenizer = tokenize
    else:
        import open_clip
        root_dir = os.getcwd().split("ProAPO")[0]
        cache_dir = os.path.join(root_dir, "autodl-tmp/cache_model")

        if model_type == "openclip":
            if model_name == "laion2b-ViT-B-32":    
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-B-32')
            elif model_name == "laion2b-ViT-B-16":
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-B-16')
            elif model_name == "laion2b-ViT-L-14":
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-L-14')
        elif model_type == "EVA02":
            if model_name == "B-16":
                model, _, preprocess = open_clip.create_model_and_transforms('EVA02-B-16', pretrained='merged2b_s8b_b131k', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('EVA02-B-16')
            elif model_name == "L-14":
                model, _, preprocess = open_clip.create_model_and_transforms('EVA02-L-14', pretrained='merged2b_s4b_b131k', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('EVA02-L-14')
        elif model_type == "SigLIP":
            if model_name == "ViT-B-16":
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')
            elif model_name == "ViT-L-16-256":
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained='webli', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-L-16-SigLIP-256')
            elif model_name == "ViT-SO400M-14":
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli', cache_dir=cache_dir, device="cuda")
                tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP')

        else:
            print("Error Model Name !", args.model)

    model.eval()

    # ---------- step1: build dataset ---------- #  
    print("\nload dataset =>")
    train_loader, test_loader, class_name_list = get_named_data(args, preprocess)
    
    # ---------- step3: iterative optimization ---------- # 
    if args.model_mode == "zero_shot":
        print("\ntest Zero-Shot result =>")
        test_clip_zero_shot(args, model, class_name_list, test_loader, tokenizer)

    else:
        print("\niterative optimization Prompt =>")
        best_prompt, exp_name = iterative_select_optimal_prompt(args, model, tokenizer, class_name_list, train_loader, test_loader, args.threshold_worst_decrease, exp_name=args.model_name)
 