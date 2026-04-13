# This file is adapted from the PEFT library
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications Copyright (c) [PlaceHolder]
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# See: 
# http://creativecommons.org/licenses/by-nc/4.0/
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
#from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
#from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from peft.utils.other import transpose

from .config import GAWSDConfig



class GAWSDLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("GAWSD",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("input_splits", "output_splits", "input_split_dim", "output_split_dim")

    def __init__(self, base_layer: nn.Module, **kwargs): 
        self.base_layer = base_layer
        self.input_splits = {}
        self.output_splits = {}
        self.input_split_dim = {}
        self.output_split_dim = {}
        self.GAWSD = nn.ParameterDict({})
        self.diag_bool = {}
        self.diag = nn.ParameterDict({})


        self._disable_adapters = False
        
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError (f"Unsupported layer type '{type(base_layer)}' encountered. Only linear layers are supported. ")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, input_splits, output_splits, input_split_dim, output_split_dim, init_weights, diag):
        
        
        assert self.in_features%input_splits == 0, \
            f'The number of input splits `input_splits = {input_splits}` should divide the layer input features size `in_features = {self.in_features}` equally.' 
        assert self.out_features%output_splits == 0, \
            f'The number of output splits `output_splits = {output_splits}` should divide the layer output features size `out_features = {self.out_features}` equally.' 

        input_split_dim = int(self.in_features/input_splits)
        output_split_dim = int(self.out_features/output_splits)

        self.input_splits[adapter_name] = input_splits
        self.output_splits[adapter_name] = output_splits
        self.input_split_dim[adapter_name] = input_split_dim
        self.output_split_dim[adapter_name] = output_split_dim
        self.diag_bool[adapter_name] = diag
        
            

        base_layer = self.get_base_layer()
        # Actual trainable parameters
        self.GAWSD[adapter_name] = nn.Parameter(base_layer.weight.new_zeros((output_splits,input_splits)))
        if diag: 
            self.diag[adapter_name] = nn.Parameter(base_layer.weight.new_zeros(1,input_splits))

        if init_weights:
            self.reset_gawsd_parameters(adapter_name, init_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_gawsd_parameters(self, adapter_name, init_weights):
        if init_weights == "none" :
            return
        else:
            if adapter_name in self.GAWSD.keys():
                if init_weights == "kaiming":
                    nn.init.kaiming_uniform_(self.GAWSD[adapter_name], a=math.sqrt(5))
                    if self.diag_bool[adapter_name]:
                        nn.init.kaiming_uniform_(self.diag[adapter_name], a=math.sqrt(5))
                        
                elif init_weights == "zero":
                    nn.init.constant_(self.GAWSD[adapter_name], 0.)
                    if self.diag_bool[adapter_name]:
                        nn.init.kaiming_uniform_(self.diag[adapter_name], a=math.sqrt(5))


    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)
 

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.GAWSD.keys():
                continue

            GAWSD = self.GAWSD[active_adapter]

            # getting the sub-batch, passing it to N_adapter layers and updating the corresponding indices of the linear
            # layer output
            
              
            sub_batch = x[sub_batch_indices_list[i]].to(GAWSD.dtype)
            seq_length = sub_batch.size(1)
            bs = sub_batch.size(0)
            if self.diag_bool[active_adapter]:

                gawsd_output = ((GAWSD*self.diag[active_adapter])@(sub_batch.view(bs,seq_length, self.input_splits[active_adapter],self.input_split_dim[active_adapter]))).view(bs,seq_length,-1)
            else:
                gawsd_output = (GAWSD@(sub_batch.view(bs,seq_length, self.input_splits[active_adapter],self.input_split_dim[active_adapter]))).view(bs,seq_length,-1)
                
            
            result[sub_batch_indices_list[i]] += gawsd_output.to(torch_result_dtype)

        return result






class Linear(nn.Module, GAWSDLayer):
    # GAWSD implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        input_splits: int = 0,
        output_splits: int = 0,
        input_split_dim: int = 0,
        output_split_dim: int = 0,
        diag: bool = False,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()
        GAWSDLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            input_splits = input_splits,
            output_splits=output_splits,
            input_split_dim=input_split_dim,
            output_split_dim = output_split_dim,
            init_weights=init_weights,
            diag = diag

        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

  



    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.GAWSD.keys():
                    continue
                GAWSD = self.GAWSD[active_adapter]
                x = x.to(GAWSD.dtype)
                seq_length = x.size(1)
                bs = x.size(0)

                if self.diag_bool[active_adapter]:
                    result = result + ((GAWSD*self.diag[active_adapter])@(x.view(bs,seq_length, self.input_splits[active_adapter],self.input_split_dim[active_adapter]))).view(bs,seq_length,-1)
                else:
                    result = result + (GAWSD@(x.view(bs,seq_length, self.input_splits[active_adapter],self.input_split_dim[active_adapter]))).view(bs,seq_length,-1)
            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "GAWSD." + rep







