
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

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType




@dataclass
class GAWSDConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GAWSDModel`].

    Args:
        input_splits(`int`): Number of input splits. It should be equal to the input feature dimension of the layer divided
        by the input split dimension.
        output_splits(`Optional[int]`): Number of output splits. It should be equal to the output feature dimension of the
        layer divided by the output split dimension.
        input_split_dim(`Optional[int]`): Input split dimension. It should be equal to the input feature dimension of the layer divided
        by the number of input splits.
        output_split_dim(`Optional[int]`): Output split dimension. It should be equal to the output feature dimension of the 
        layer divided by the number of output splits.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. Only linear layers are supported. If this is specified,
            only the modules with the specified names will be replaced. When passing a string, a regex match will be
            performed. When passing a list ofstrings, either an exact match will be performed or it is checked if the
            name of the module ends with any of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D 
            modules are chosen, excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for the layer. Can be 'none' or 'all' or 'gawsd_only'. If 'all' or 'gawsd_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            How to initialize the weights of the adapter layers. Passing `'True'` (default) results in the 
            kaiming initalization. Setting this argument to `'False'` leads to random initialization. 
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        input_splits_pattern (`dict`):
            The mapping from layer names or regexp expression to input splits which are different from the default input splits specified by `input splits`.
        output_splits_pattern (`dict`):
            The mapping from layer names or regexp expression to output splits which are different from the default output splits specified by `output splits`.


    """
    input_splits: int = field(
        default=0,
        metadata={
            "help": 
                ("Number of input splits. It should be equal to the input feature dimension of the layer divided by the input split dimension."),
        },
    ) 
    output_splits: Optional[int] = field(
        default=0,
        metadata={
            "help":
            ("Number of output splits. It should be equal to the output feature dimension of the layer divided by the output split dimension."),
        },
    )
    input_split_dim: Optional[int] = field(
        default=0,
        metadata={
            "help":
            ("Input split dimension. It should be equal to the input feature dimension of the layer divided by the number of input splits."),
        },
    )
    output_split_dim: Optional[int] = field(
        default=0,
        metadata={
            "help":
                  ("Output split dimension. It should be equal to the output feature dimension of the layer divided by the number of output splits."),
        },
    )

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with GAWSD."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
   
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "gawsd_only"] = field(
        default="none", metadata={"help": "Bias type for GAWSD. Can be 'none', 'all' or 'gawsd_only'"}
    )

    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from GAWSD layers to be set as trainable and saved in the final checkpoint. "
        },
    )
    diag: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add diagonal matrix to GAWSD adapter"
            ),
        },
    )


    init_weights: Literal["zero", "kaiming", "none"] = field(
        default="none", metadata={"help": (
                                  "How to initialize the weights of the GAWSD layers. Passing `'none'` (default)"
                                  "leads to completely random initialization, Passing `'kaiming'` leads to kaiming initalization."
                                    " Passing `'zero'` leads to zero initialization. "
        ),
                                 },
    )
    
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "List of layer indices to be transformed by GAWSD. If not specified, all layers in target_modules are transformed "
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "Pattern to match layer names in target_modules, if layers_to_transform is specified. By default PeftModel will look at common layer pattern (layers, h, blocks, etc.), use it for exotic and custom models."
            "This only works when target_modules is a list of str."
            
        },
    )
    input_splits_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to input splits which are different from the default input splits specified by `input splits`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    output_splits_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to output splits which are different from the default output splits specified by `output splits`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )



    def __post_init__(self):
        self.peft_type = PeftType.GAWSD
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")
        

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # input splits should be positive. output splits should be specified if input splits is positive.
        if self.input_splits>0:
            if self.output_splits == 0:
                self.output_splits = self.input_splits
        else:
            raise ValueError(f"`input_splits` should be a positive integer value but the value passed is {self.input_splits}")

        if self.input_splits_pattern and not self.output_splits_pattern:
            self.output_splits_pattern = self.input_splits_pattern
            


      

 


