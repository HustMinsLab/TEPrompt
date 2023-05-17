# TEPrompt
This repository provides the code of TEPrompt model for the ACL 2023 paper: TEPrompt: Task Enlightenment Prompt Learning for Implicit Discourse Relation Recognition.

# Data
We use the PDTB 3.0 corpus for evaluation. Due to the LDC policy, we cannot release the PDTB data. If you have bought data from LDC, please put the PDTB .tsv file in dataset.

# Requirements
python 3.7.13 
torch == 1.11.0  
transformers == 4.15.0

# How to use
- You have to put the PDTB corpus file in dataset file first.
- For each Pre-trained Language Model (PLM), run file
```
python main.py
```

# Citation
Please cite our paper if you use the code!
