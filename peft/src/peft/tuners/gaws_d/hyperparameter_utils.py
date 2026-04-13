# Copyright (c) [PlaceHolder]
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# See: 
# http://creativecommons.org/licenses/by-nc/4.0/
from transformers.pytorch_utils import Conv1D
import torch.nn as nn
import pandas as pd




def get_valid_splits_values(model, target_module):
    for name,param in model.named_modules():
        if target_module in name:
            splitted_name = name.split('.')
            layer = recursive_getattr(model, splitted_name)
            if isinstance(layer, nn.Linear):
                in_features, out_features = layer.in_features, layer.out_features
            elif isinstance(layer, Conv1D):
                in_features, out_features = (layer.weight.ds_shape if hasattr(layer.weight, "ds_shape") else layer.weight.shape)
            else:
                raise ValueError (f"Unsupported layer type '{type(layer)}' encountered. Only linear layers are supported. ")
            
            valid_split_dim = get_factors(in_features, out_features)
            return valid_split_dim

        
    raise ValueError (f"The specified target_module = {target_module} is not found in the passed model. Please ensure that the module name is correct and exists within the model's architecture.")
            

   
def recursive_getattr(model,splitted_name):
    obj = model
    for attribute_name in splitted_name:
        obj = getattr(obj,attribute_name)
    return obj





def get_factors(in_features, out_features):
    const = out_features/in_features
    input_split_dim = []
    input_splits = []
    output_splits = []
    for factor in range(2, in_features):
        if in_features % factor ==  0:
            input_split_dim.append(factor)
            input_splits.append(int(in_features /factor))
            output_splits.append(int(const*in_features /factor))

    df_dic = {
        "input_splits": input_splits,
        "output_splits": output_splits,
        "input_splits_dim = output_splits_dim": input_split_dim,
    }

    df = pd.DataFrame(df_dic)

    return df
           