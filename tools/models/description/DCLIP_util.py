# ==================== Import Packages ==================== #
import time
import sys
import os 


import numpy as np 
import json 


# ==================== Functions ==================== #
def make_descriptor_sentence(descriptor):

    descriptor = descriptor.lower()

    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"
