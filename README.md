

## Repository Structure


```commandline
.
└── R0_Structured_Information_Extraction                        <- root directory of the repository
    ├── data
    │   ├── processed                                           <- contains the processed data
                      │   ├── dev_train_dev_test 
                      │   ├── test_train_dev_test
                      │   ├── train_train_dev_test
                      │   ├── final_datasets
                                                   
    │   └── raw  
    │                 │   ├── cord19_train_dev_test             <- contains the raw data
    │                 │   ├── initial_datset                   
    │
    ├── experimental_results                                    <- contains obtained results from experiments
    ├── models                                                  <- contains the trained/used models (empty now as models are too big to be uploaded)
    ├──src                                                      <- contains the source code/scripts
    ├──test
    ├── README.md                                               <- README file for documenting the service.
    ├── requirements.txt                                        <- contains python requirements listed with specifying the versions
```

## R0_Structured_Information_Extraction
This repository contains the dataset and scripts for the research paper [Large Language Models for Scientific Information Extraction: An Empirical Study for Virology](https://arxiv.org/abs/2401.10040).
This work aims to provide models that extract structured information from the title/abstract combinations in the field of virology that investigate the "Basic Reproduction Number" aka "R0 number" of infectious diseases. Our models are trained to extract six salient properties which are a sufficient set of properties to summarise one contribution in virology addressing this research problem, considering the semantic model used to produce the relevant comparison at [https://orkg.org/comparison/R44930/](https://orkg.org/comparison/R44930/).

Given this introduction, the models trained in this work are capable of receiving a title/abstract combination of a scholarly text in virology that studies the R0 value of infectious disease, and then producing a structured summary highlighting the "disease name", "location", "date", "r0 value", "%ci-values", and "method". Some input texts might contain more than one contribution, in this case the model aims to report all the contributions that can be found in a text. On the other hand, some text might just contain some related keywords to the R0 number but not investigate or report a real value, in this case, the model aims to say "unanswerable".


To address this objective we first created a dataset available in this repository under the [data directory](/data) and also on Zenodo at [https://zenodo.org/records/10003640](https://zenodo.org/records/10003640).

With this dataset, we instruction fine-tuned [Flan-T5 large](https://huggingface.co/google/flan-t5-large) using drop and squad_v2 instruction collections from FLAN instructions available at: https://github.com/google-research/FLAN/blob/main/flan/templates.py.

We finetuned 40 models overall using different combinations of instruction templates from drop and squad_v2, some models only with one instruction, some with multiple (best combinations), and also with all instructions. Other than the templates, one category of our models produces just a structured text as a result while the other category is trained to produce a valid JSON string as their response. The best model from the category of JSON generators is available at Huggingface  [R0_contribution_IE ](https://huggingface.co/orkg/R0_contribution_IE).

### dataset
To create the dataset of this work, the [cord-19 collection](https://github.com/allenai/cord19) was used as the initial corpus. We created the final dataset of r0-contributions using the scripts under [/src/data](/src/data) and the process of manual annotation. the resulting dataset is located at [data/raw/cord19_train_dev_test](data/raw/cord19_train_dev_test) in Excel format. As we aimed to instruction finetune the models, we further fed this raw data into the templates and created json datasets available under [data/processed](data/processed). Other than JSON files, we built the datasets in arrow format, which is more convenient to work with. The datasets in arrow format are available under [data/processed/final_datasets](data/processed/final_datasets), each directory inside this path contains a dataset, and the training set is based on the template in the folder name.

### fine-tuning scripts and resulting models

### evaluation scripts



