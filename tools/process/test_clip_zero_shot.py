# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch 
import clip 

from tqdm import tqdm

from models.util_prompt import load_prompt_classes_agnostic, get_prompt, get_prompt_with_description, get_full_prompt_from_json_path
from process.select_optimal_prompt import get_image_features
from utils.util_model import accuracy
from sklearn.metrics import classification_report, accuracy_score


# ==================== Functions ==================== #
def test_clip_zero_shot(args, model, class_name_list, test_loader, tokenizer):

    prompt_template_list = load_prompt_classes_agnostic(args, args.mode_prompt_model_agnostic, args.path_prompt_model_agnostic)
    prompt_list = get_prompt(args, prompt_template_list, class_name_list, args.path_prompt_synonym_label, args.add_dataset_species_flag, args.dataset.lower())

    model_name_str = args.model.replace("/", "_").replace("@", "_")
    test_label, test_feature = get_image_features(args, test_loader, model, mode=f"{model_name_str}_test")

    with torch.no_grad():
        for idx, temp_class_prompts in tqdm(enumerate(prompt_list)):
            temp_prompts = tokenizer(temp_class_prompts).cuda() 
            temp_text_features = model.encode_text(temp_prompts)

            if idx == 0:
                text_features = temp_text_features
            else:
                text_features = torch.cat((text_features, temp_text_features), dim=0)

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * test_feature @ text_features.t()

        output = logits.softmax(dim=-1)

        prec1, prec5 = accuracy(output.data, test_label, topk=(1, 5))

        print(f"prec1: {prec1}, prec5: {prec5}")

        