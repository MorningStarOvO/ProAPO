# ==================== Import Packages ==================== #
import time
import sys
import os 
import shutil
import psutil

import numpy as np 
import json 


# ==================== Functions ==================== #


def save_json(dict, path_save):

    data_json = json.dumps(dict, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()


def sorted_dict(dict, reverse=True):

    dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=reverse)

    dict_new  = {}
    for key in dict_sorted:
        dict_new[key[0]] = key[1]

    return dict_new
    
def sorted_dict_by_key(dict, key, reverse=True):

    dict_sorted = sorted(dict.items(), key = lambda i: i[1][key], reverse=reverse)

    dict_new  = {}
    for key in dict_sorted:
        dict_new[key[0]] = key[1]

    return dict_new
    

def sorted_dict_and_save_json(dict, path_save, reverse=True):

    dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=reverse)

    dict_new  = {}
    for key in dict_sorted:
        dict_new[key[0]] = key[1]


    data_json = json.dumps(dict_new, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()

    return dict_new

def build_save_file(args):

    path_dataset_output = os.path.join(args.checkpoint_path, args.dataset, "task_output", args.task_name)
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output, exist_ok=True)
    args.path_dataset_output = os.path.join(args.checkpoint_path, args.dataset)


    path_task_save = os.path.join(args.checkpoint_path, args.dataset, "task")
    if not os.path.exists(path_task_save):
        os.makedirs(path_task_save, exist_ok=True)


    path_log_save = os.path.join(path_task_save, "log", args.task_name)
    if not os.path.exists(path_log_save):
        os.makedirs(path_log_save, exist_ok=True)
    args.path_log_save = path_log_save


    path_analysis_save = os.path.join(path_task_save, "analysis", args.task_name)
    if not os.path.exists(path_analysis_save):
        os.makedirs(path_analysis_save, exist_ok=True)
    args.task_analysis_dir = path_analysis_save


    path_best_prompt_save = os.path.join(path_task_save, "best_prompt")
    args.path_best_prompt_save = path_best_prompt_save
    if not os.path.exists(path_best_prompt_save):
        os.makedirs(path_best_prompt_save, exist_ok=True)
    

    path_code_save = os.path.join(path_dataset_output, "code")
    if not os.path.exists(path_code_save):
        os.makedirs(path_code_save, exist_ok=True)


    if not args.debug:   
        task_dir = os.path.join(path_task_save, f"{args.task_name}.json")
        args.task_dir = task_dir

        task_prompt_dir = os.path.join(path_best_prompt_save, f"{args.task_name}")
        args.task_prompt_dir = task_prompt_dir


    model_name = args.model_name

    
    if not args.debug:        

        code_dir = path_code_save

        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        shutil.copytree("tools", code_dir)
        
        get_command_test()

def judge_exp_has_been_run(args):
    
    if not os.path.exists(args.task_dir):
        return 0
    else:
        with open(args.task_dir, 'r') as f:
            data_task_json = json.load(f)

        if args.model_name is not None:
            key_task = args.model_name.replace(args.task_name, "")

            if key_task in data_task_json:
                print("The experiment is complete !")
                sys.exit()

def get_command_test():

    # get PID # 
    pid = os.getpid()

    s = psutil.Process(pid)

    command_text = "" 
    for word in s.cmdline():
        command_text += word 
        command_text += " "
    
    print(f"\t{command_text}")


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def exit(self):
        self.log.close()
