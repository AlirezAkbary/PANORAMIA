# PANORAMIA Privacy Auditing Pipeline
This repository contains the implementation of the privacy auditing pipeline described in the NeurIPS 2024 paper _PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining_ (https://arxiv.org/abs/2402.09477) on Large Language Models.

## Installation
* Create a virtual environment using python 3.

`virtualenv -p python3 panoramia_venv`

* Activate the virtual environment.

`source panoramia_venv/bin/activate`

* Clone the repo, and install the necessary python packages with `requirements.txt` file.

`pip install -r requirements.txt`

## Running the code
To run the whole pipeline of PANORAMIA, from training the target model to get the audit values, it might be too long. Here, we explain how you can run each module separately. Later, we explain how you can run the whole pipeline all at once if you want to. 

### Training the Generative Model

```python
python -m src.main --base_train_load_generator \
                   --base_log_dir \
                   --generator_train_pretrained_model_name_or_path \
                   --generator_train_saving_dir 
                   
```
The trained generator with the least validation loss will be saved to `outputs/generator/saved_model/checkpoint-XXXX/`, where `XXXX` is a checkpoint number.


### Generating Synthetic Samples
To run the next step using the saved checkpoint of the generator model, you can retrieve the checkpoint directory automatically by running:
```bash
GEN_CHECKPOINT_DIR=$(ls -td outputs/generator/saved_model/checkpoint-* | head -1)
```

```python
python -m src.main --base_train_load_generator \
                   --base_generate_samples \
                   --base_log_dir \ 
                   --generator_train_pretrained_model_name_or_path \
                   --generator_train_saving_dir \
                   --generator_generation_saving_dir
                   
```


### Training the Target Model

```python
python -m src.main --base_train_load_target \
                   --base_log_dir \
                   --audit_target_saving_dir \
                   --audit_target_pretrained_model_name_or_path \
                   --
```
this will result in multiple checkpoints throught the training of the target models. Choose a checkpoint for which the model privacy would be audited. make sure enough space exists as there are too many checkpoints. 

note: in saving, the epoch also affects the address of the saved target model -> take care of that

```bash
TARGET_CHECKPOINT=
```

### Training the Helper Model
```python
python -m src.main --base_train_load_helper \
                   --base_log_dir \
                   --dataset_path_to_synthetic_data \
                   --audit_helper_saving_dir \
                   --audit_helper_pretrained_model_name_or_path \
```  

note: in saving, the epoch also affects the address of the saved helper model -> take care of that

also, find out the checkpoint of the helper

```bash
HELPER_CHECKPOINT=
```

### Training the Baseline Classifier and Saving its Predicitions on the Evaluation Set

```python
python -m src.main  --base_log_dir \
                    --base_train_load_helper \
                    --base_train_baseline \    
                    --dataset_path_to_synthetic_data \
                    --dataset_mia_num_train \
                    --dataset_mia_num_val \
                    --dataset_mia_num_test \
                    --audit_helper_saving_dir \
                    --attack_baseline_training_args_output_dir  
                    
```
<!-- base_attack_main argument has been deleted. Take care of it -->
This will leave blah blah blah

### Training the MIA Classifier and Saving its Predicitions on the Evaluation Set

```python
python -m src.main  --base_log_dir \
                    --base_train_load_target \
                    --base_train_mia \ 
                    --dataset_path_to_synthetic_data \
                    --dataset_mia_num_train \
                    --dataset_mia_num_val \
                    --dataset_mia_num_test \
                    --audit_target_saving_dir \
                    --attack_mia_training_args_output_dir
```

## Project Structure
