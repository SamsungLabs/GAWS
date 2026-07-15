<h1 align="center">
  <p>
    Grouped Adaptive Weight Sharing (GAWS): An Inference-Efficient Adaptation Method for Large Language Models
  </p>
</h1>

<p align="left">
  <b> This work has been accepted to ACL 2026 [Findings].</b>
</p>

<p align="left">
  <a href="https://aclanthology.org/2026.findings-acl.1590/">
    📝 Paper
  </a>
</p>

---

## 📌 Overview

Although **Low-Rank Adaptation (LoRA)** has revolutionized parameter-efficient fine-tuning (PEFT), it introduces additional inference overhead due to the extra computation required by adapter layers. Most existing approaches focus primarily on improving accuracy or reducing the number of trainable parameters, while inference efficiency—especially in the **unmerged adapter setting**—remains underexplored.

In this work, we study the inference bottlenecks of LoRA adapters and identify **segmented function calls and adapter execution overhead** as major contributors to latency on GPUs. To address this issue, we propose:

> **Grouped Adaptive Weight Sharing (GAWS)**, an inference-efficient adapter design based on structured Kronecker product decomposition.

GAWS reduces runtime overhead while preserving the parameter efficiency and adaptation capability of LoRA. Extensive experiments on **T5-3B, GPT-2 Large, LLaMA 3.2-3B, and RoBERTa-Large** demonstrate that GAWS achieves a strong accuracy–latency trade-off.

Specifically, GAWS reduces inference latency to approximately **40% of the latency gap between unmerged LoRA and the base model**, making it a Pareto-efficient solution for deploying adapted large language models in latency-sensitive environments.

---


## 🚀 Quickstart

### 📥 Installation

Clone the repository and install the modified PEFT library:

```bash
git clone <insert_github_repository>
cd GAWS/peft
pip install -r requirements.txt
pip install .
```



### 🔧 Using GAWS

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


---
##  📚 Citation

If you find GAWS useful in your research, please cite:

```bash
@inproceedings{alsuradi-etal-2026-grouped,
    title = "Grouped Adaptive Weight Sharing ({GAWS}): An Inference-Efficient Adaptation Method for Large Language Models",
    author = "Alsuradi, Eman  and
      Lee, Junhyun  and
      Lee, Kyenghun  and
      Ko, Hyeonmok  and
      Jubair, Fahed",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1590/",
    doi = "10.18653/v1/2026.findings-acl.1590",
    pages = "31790--31806",
    ISBN = "979-8-89176-395-1",
}
```

---

## 📬 Contact

For questions, discussions, or bug reports, please open an issue in this repository or contact at eman.zaki@samsung.com.
