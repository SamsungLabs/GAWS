<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p> GAWS: Grouped Adaptive Weight Sharing</p></h1>

</h3>

## Description

Adapter-based parameter-efficient fine-tuning enables multitask learning by attaching lightweight, task-specific adapters to a shared base model. However, efficiently serving multiple adapters poses deployment challenges. While merging adapters into the base model eliminates runtime overhead, it hinders model sharing across tasks, introduces potential numerical instability on quantized models, and complicates deployment in environments with static computational graphs. Conversely, serving unmerged adapters avoids these issues but comes at the cost of increased inference latency. Through analysis of LoRA adapters on GPUs, we attribute this latency primarily to segmented function calls.  To address this, we propose Grouped Adaptive Weight Sharing (GAWS), a novel adapter design based on structured Kronecker product decomposition. Experiments on T5, GPT-2 Large, and LLaMA-3B show that GAWS reduces latency to about 42\% of the gap between LoRA and the base model, while maintaining parameter efficiency and comparable accuracy.  This positions GAWS as an effective solution for efficient multitask deployment.

## Quickstart

Clone the repository and install the peft library:

```bash
git clone <insert_github_repository>
cd GAWS/peft
pip install -r requirements.txt
pip install .
```

Prepare a model for training with GAWS by wrapping the base model and GAWS configuration with `get_peft_model`.

```python
from transformers import AutoModelForCausalLM
from peft import  GAWSConfig, PeftModel, get_peft_model # for GAWS-D use from peft import  GAWSDConfig
from peft.tuners.gaws import get_valid_split_dim_values # for GAWS-D use from peft.tuners.gaws_d import get_valid_splits_values

# load the base model
model_name_or_path = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# to get valid split dimension values
get_valid_split_dim_values(model, "q_proj") 

"""
    input_split_dim  output_split_dim  input_splits = output_splits
0              1280              1280                             2
1               640               640                             4
2               512               512                             5
3               320               320                             8
4               256               256                            10
5               160               160                            16
6               128               128                            20
7                80                80                            32
8                64                64                            40
9                40                40                            64
10               32                32                            80
11               20                20                           128
12               16                16                           160
13               10                10                           256
14                8                 8                           320
15                5                 5                           512
16                4                 4                           640
17                2                 2                          1280
"""


# construct adapter config
config = GAWSConfig(
    task_type="CAUSAL_LM",
target_modules=["v_proj", "q_proj", 'k_proj'],
    input_splits = 10,
    output_splits = 10,
    input_split_dim = 256,
    output_split_dim = 256,
    init_weights = "zero", # There are 3 available options 1) "zero" : zero initialization 2) "kaiming": kaiming initialization 3) "none": random initialization
    diag = False # Whether to add diagonal matrix to the model
    
)
# construct GAWS model
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 6,291,456 || all params: 2,785,975,296 || trainable%: 0.2258"
```
To save the GAWS model: 
```python
model.save_pretrained('phi2_gaws-s256')
```

To load the GAWS model for inference:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained("phi2_gaws-s256").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough on a baking sheet. Bake for 10-12 minutes or until golden brown."

```

## Contact

If you have any questions, please create an issue on this repository or contact at []

