# ==================== Import Packages ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from tqdm import tqdm  

import torch 
import clip 
import torch.nn.functional as F

from einops import rearrange, repeat

from utils.util_model import accuracy
from utils.util import save_json, sorted_dict, sorted_dict_by_key


# ==================== Functions ==================== #
def get_redundant_matrix_score_l2_norm(features):

    similar_score = features @ features.t()

    I = torch.eye(similar_score.shape[0]).cuda()
    similar_score = torch.triu(similar_score - I)

    redundant_score = torch.norm(similar_score, p=2)

    return redundant_score


def get_redundant_matrix_score_compare_most_similar_feature(features):
    
    similar_score = features @ features.t()

    I = torch.eye(similar_score.shape[0]).cuda()
    most_similar_score, _ = (similar_score - I).max(dim=0)

    redundant_score = torch.sum(torch.ones_like(most_similar_score).cuda() - most_similar_score)

    return redundant_score

def get_logits_compare_most_similar_score(logits, labels):
    
    true_label_logits = logits[range(labels.shape[0]), labels].detach().clone()
    true_label_logits = repeat(true_label_logits, 'a -> a b', b=logits.shape[1])
    
    logits[range(labels.shape[0]), labels] = 0
    
    max_score, _ = (logits - true_label_logits).max(dim=0)

    return max_score.sum()


def get_acc_scores(image_features, labels, text_features, model, return_detail_flag=False, analysis_flag=False):
    
    extra_info = {}

    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        output = logits.softmax(dim=-1)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        
        label_test_list = torch.unique(labels).detach().cpu().numpy().tolist()
        # print(len(label_test_list))
        acc_per_class = torch.zeros(len(label_test_list))
        acc_5_per_class = torch.zeros(len(label_test_list))
        score_5, pred_5 = output.topk(5, 1, True, True)

        pred_5 = pred_5.t()
        correct_5 = pred_5.eq(labels.view(1, -1).expand_as(pred_5))

        average_per_class_top_1_acc = acc_per_class.mean() * 100 #
        average_per_class_top_5_acc = acc_5_per_class.mean() * 100 #

        avg_acc = round(average_per_class_top_1_acc.item(), 4)
        avg_acc_5 = round(average_per_class_top_5_acc.item(), 4)
        acc_top1 = round(prec1.item(), 4)
        acc_top5 = round(prec5.item(), 4)

        # print("avg_acc: ", avg_acc)
        # print("avg_acc_5: ", avg_acc_5)
        # print("acc_top1: ", acc_top1)
        # print("acc_top5: ", acc_top5)

        if return_detail_flag:
            extra_info["acc_per_class"] = acc_per_class

        if analysis_flag:

            # ----- 计算 text-to-text inter-class cosine similarity ----- # 
            redundant_score_l2_norm = get_redundant_matrix_score_l2_norm(text_features).item()

            # ----- 计算 text-to-image inter-class similarity ----- # 
            inter_class_score_CE = F.cross_entropy(logits, labels).item()

            extra_info["redundant_score_l2_norm"] = redundant_score_l2_norm
            # extra_info["redundant_score_most_similar"] = redundant_score_most_similar
            extra_info["inter_class_score_CE"] = inter_class_score_CE
            # extra_info["inter_class_score_most_similar"] = inter_class_score_most_similar
            

        if return_detail_flag or analysis_flag:
            return avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info
        else:
            return avg_acc, avg_acc_5, acc_top1, acc_top5


def get_acc_scores_based_on_logits(text_features, logits, labels, model, return_detail_flag=False, analysis_flag=False):
    
    
    extra_info = {}

    with torch.no_grad():
        
        output = logits.softmax(dim=-1)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        
        label_test_list = torch.unique(labels).detach().cpu().numpy().tolist()
        # print(len(label_test_list))
        acc_per_class = torch.zeros(len(label_test_list))
        acc_5_per_class = torch.zeros(len(label_test_list))
        score_5, pred_5 = output.topk(5, 1, True, True)

        pred_5 = pred_5.t()
        correct_5 = pred_5.eq(labels.view(1, -1).expand_as(pred_5))

        # for i in label_test_list:
        #     idx = (labels == i)
        #     acc_per_class[i] = torch.sum(correct_5[0, idx]).item() / torch.sum(idx).item()
        #     acc_5_per_class[i] = torch.sum(correct_5[:, idx]).item() / torch.sum(idx).item()

        average_per_class_top_1_acc = acc_per_class.mean() * 100 #
        average_per_class_top_5_acc = acc_5_per_class.mean() * 100 #

        avg_acc = round(average_per_class_top_1_acc.item(), 4)
        avg_acc_5 = round(average_per_class_top_5_acc.item(), 4)
        acc_top1 = round(prec1.item(), 4)
        acc_top5 = round(prec5.item(), 4)

        # print("avg_acc: ", avg_acc)
        # print("avg_acc_5: ", avg_acc_5)
        # print("acc_top1: ", acc_top1)
        # print("acc_top5: ", acc_top5)

        if return_detail_flag:
            extra_info["acc_per_class"] = acc_per_class

        if analysis_flag:

            # ----- 计算 text-to-text inter-class cosine similarity ----- # 
            redundant_score_l2_norm = get_redundant_matrix_score_l2_norm(text_features).item()
            # redundant_score_most_similar = get_redundant_matrix_score_compare_most_similar_feature(text_features).item()
            # print("acc_top1: ", acc_top1)
            # print("redundant_score_l2_norm: ", redundant_score_l2_norm)
            # print("redundant_score_most_similar: ", redundant_score_most_similar)

            # ----- 计算 text-to-image inter-class similarity ----- # 
            inter_class_score_CE = F.cross_entropy(logits, labels).item()
            # inter_class_score_most_similar = get_logits_compare_most_similar_score(output, labels).item()
            # print("inter_class_score_CE: ", inter_class_score_CE)
            # print("inter_class_score_most_similar: ", inter_class_score_most_similar)

            extra_info["redundant_score_l2_norm"] = redundant_score_l2_norm
            # extra_info["redundant_score_most_similar"] = redundant_score_most_similar
            extra_info["inter_class_score_CE"] = inter_class_score_CE
            # extra_info["inter_class_score_most_similar"] = inter_class_score_most_similar
            

        if return_detail_flag or analysis_flag:
            return avg_acc, avg_acc_5, acc_top1, acc_top5, extra_info
        else:
            return avg_acc, avg_acc_5, acc_top1, acc_top5

def get_acc_scores_and_dict(class_name_list, image_features, labels, text_features, model, threshold_confuse):
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        output = logits.softmax(dim=-1)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        
        label_test_list = torch.unique(labels).detach().cpu().numpy().tolist()
        acc_per_class = torch.zeros(len(label_test_list))
        acc_5_per_class = torch.zeros(len(label_test_list))
        score_5, pred_5 = output.topk(5, 1, True, True)

        pred_5 = pred_5.t()
        correct_5 = pred_5.eq(labels.view(1, -1).expand_as(pred_5))

        acc_dict = {} 
        acc_error_dict = {} 

        for i in label_test_list:

            temp_category = class_name_list[i]

            idx = (labels == i)
            acc_per_class[i] = torch.sum(correct_5[0, idx]).item() / torch.sum(idx).item()
            acc_5_per_class[i] = torch.sum(correct_5[:, idx]).item() / torch.sum(idx).item()

            acc_dict[temp_category] = {}
            acc_dict[temp_category]["top1"] = round(acc_per_class[i].item()*100, 2)
            acc_dict[temp_category]["top5"] = round(acc_5_per_class[i].item()*100, 2)

            acc_error_dict[temp_category] = {}
            acc_error_dict[temp_category][temp_category] = round(acc_per_class[i].item()*100, 2)
            for j in label_test_list:
                if j != i:
                    error_acc = torch.sum(pred_5[0, idx] == j).item() / torch.sum(idx).item()
                    error_acc = round(error_acc*100, 2)

                    if error_acc > threshold_confuse:
                        acc_error_dict[temp_category][class_name_list[j]] = error_acc

            acc_error_dict[temp_category] = sorted_dict(acc_error_dict[temp_category])


        acc_dict = sorted_dict_by_key(acc_dict, key="top1", reverse=False)
        acc_error_dict_sorted = {}
        for temp_category in acc_dict:
            acc_error_dict_sorted[temp_category] = acc_error_dict[temp_category]

        average_per_class_top_1_acc = acc_per_class.mean() * 100 
        average_per_class_top_5_acc = acc_5_per_class.mean() * 100 

        avg_acc = round(average_per_class_top_1_acc.item(), 4)
        avg_acc_5 = round(average_per_class_top_5_acc.item(), 4)
        acc_top1 = round(prec1.item(), 4)
        acc_top5 = round(prec5.item(), 4)

        # print("avg_acc: ", avg_acc)
        # print("avg_acc_5: ", avg_acc_5)
        # print("acc_top1: ", acc_top1)
        # print("acc_top5: ", acc_top5)

        return avg_acc, avg_acc_5, acc_top1, acc_top5, acc_dict, acc_error_dict_sorted


def get_acc_scores_and_dict_quick_and_group(class_name_list, image_features, labels, text_features, model, 
                                            threshold_confuse, threshold_group_mode, max_consider,
                                            return_detail_flag=False, analysis_flag=False):
    
    extra_info = {}

    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        output = logits.softmax(dim=-1)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        
        label_test_list = torch.unique(labels).detach().cpu().numpy().tolist()
        # print(len(label_test_list))
        acc_per_class = torch.zeros(len(label_test_list))
        acc_5_per_class = torch.zeros(len(label_test_list))
        score_5, pred_5 = output.topk(5, 1, True, True)

        pred_5 = pred_5.t()
        correct_5 = pred_5.eq(labels.view(1, -1).expand_as(pred_5))

        acc_top1 = round(prec1.item(), 2)
        acc_top5 = round(prec5.item(), 2)

        if threshold_group_mode == "under_baseline":
            threshold_group_acc = acc_top1
        else:
            threshold_group_acc = 100

        acc_dict = {} 

        for i in label_test_list:

            temp_category = class_name_list[i]

            idx = (labels == i)
            temp_idx_sum = torch.sum(idx).item()
            acc_per_class[i] = torch.sum(correct_5[0, idx]).item() / temp_idx_sum
            acc_5_per_class[i] = torch.sum(correct_5[:, idx]).item() / temp_idx_sum

            acc_dict[temp_category] = {}
            acc_dict[temp_category]["top1"] = round(acc_per_class[i].item()*100, 2)
            acc_dict[temp_category]["top5"] = round(acc_5_per_class[i].item()*100, 2)
            acc_dict[temp_category]["inter_class_score_CE"] = F.cross_entropy(logits[idx], labels[idx]).item()

        acc_dict = sorted_dict_by_key(acc_dict, key="inter_class_score_CE", reverse=False)

        category_select_flag = {}
        for temp_category in class_name_list:
            category_select_flag[temp_category] = 0
        category_group_split = {} 
        group_index = 0
        num_consider = 0

        for i, temp_category in enumerate(acc_dict):

            if category_select_flag[temp_category]:
                continue 

            idx = (labels == i)
            
            category_group_split[str(group_index)] = [temp_category]
            category_select_flag[temp_category] = 1
            num_consider += 1

            temp_idx_sum = torch.sum(idx).item()

            for j in label_test_list:
                if j != i:
                    error_acc = torch.sum(pred_5[:, idx] == j).item() / temp_idx_sum
                    error_acc = round(error_acc*100, 2)

                    if error_acc >= threshold_confuse:
                        category_group_split[str(group_index)].append(class_name_list[j])
                        category_select_flag[class_name_list[j]] = 1
                        num_consider += 1

            group_index += 1

            if num_consider >= max_consider and max_consider > 0:
                break 
        
        average_per_class_top_1_acc = acc_per_class.mean() * 100
        average_per_class_top_5_acc = acc_5_per_class.mean() * 100

        avg_acc = round(average_per_class_top_1_acc.item(), 4)
        avg_acc_5 = round(average_per_class_top_5_acc.item(), 4)
        
        if return_detail_flag:
            extra_info["acc_per_class"] = acc_per_class

        if analysis_flag:

            redundant_score_l2_norm = get_redundant_matrix_score_l2_norm(text_features).item()
            
            inter_class_score_CE = F.cross_entropy(logits, labels).item()
            
            extra_info["redundant_score_l2_norm"] = redundant_score_l2_norm
            # extra_info["redundant_score_most_similar"] = redundant_score_most_similar
            extra_info["inter_class_score_CE"] = inter_class_score_CE
            # extra_info["inter_class_score_most_similar"] = inter_class_score_most_similar

        # print("avg_acc: ", avg_acc)
        # print("avg_acc_5: ", avg_acc_5)
        # print("acc_top1: ", acc_top1)
        # print("acc_top5: ", acc_top5)

        # print("category_group_split: ", category_group_split)
            
        if return_detail_flag or analysis_flag:
            return avg_acc, avg_acc_5, acc_top1, acc_top5, acc_dict, category_group_split, extra_info
        else:
            return avg_acc, avg_acc_5, acc_top1, acc_top5, acc_dict, category_group_split
