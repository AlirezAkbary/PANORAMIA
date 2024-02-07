import logging
import os
import sys
from typing import Optional
import random

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer



class PANORAMIADataModule:
    """
    PANORAMIADataModule Intializes dataloaders, datasets for all the experiments
    
    DataHandler manages providing appropriate training, auditing, etc dataset for
    different parts of the PANORAMIA pipeline.

    DataHandler should satisfy consistency  throughout the PANORAMIA pipeline.
    For example, after fixing a seed, the synthetic samples chosen for beinig auditing
    examples should be the same in the baseline and the mia.

    Should DataHandler support different datasets except from WT-2 and WT-103?

    Args:

    """
    def __init__(self,
        path: str,
        name: Optional[str] = None,
        path_to_synthetic_data: str = None,
        synthetic_text_column_name: str = 'text',
        seed:  Optional[int] = None,
        do_shuffle: bool = False,
        pretrained_model_name_or_path: str = 'gpt2',
        block_size: int = 128,
        generator_train_percent: int = 25,
        prompt_sampling_percent: int = 13,
        target_model_percent: int = 50,
        helper_model_percent: int = 50,
        helper_model_train_data_mode: str = 'syn',
        syn_audit_percent: int = 30,
        mia_num_train : int = 500,
        mia_num_val : int = 500,
        mia_num_test : int = 1000,
        include_synthetic: bool = False,
        audit_mode: str = '',
        num_syn_canary: int = 2000,
        game_seed: int = 30,
        include_auxilary: bool = False,
        num_aux_in: int = 10000,
        combine_wt2_test: bool = False,
        attach_id: bool = False
    ) -> None:
        """
        Parameters
        ----------
        helper_model_percent: int
            allocates the target_model_percent%:(target_model_percent+helper_model_percent)% of the original dataset for training of the helper model.
        """
        self.path = path
        self.name = name
        self.path_to_synthetic_data = path_to_synthetic_data
        self.synthetic_text_column_name = synthetic_text_column_name
        self.seed = seed,
        self.do_shuffle = do_shuffle
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.block_size = block_size
        self.generator_train_percent = generator_train_percent
        self.prompt_sampling_percent = prompt_sampling_percent
        self.target_model_percent = target_model_percent
        self.helper_model_percent = helper_model_percent
        self.helper_model_train_data_mode = helper_model_train_data_mode
        self.syn_audit_percent = syn_audit_percent
        self.mia_num_train = mia_num_train
        self.mia_num_val = mia_num_val
        self.mia_num_test = mia_num_test # m in theory
        self.include_synthetic = include_synthetic
        self.audit_mode = audit_mode
        self.num_syn_canary = num_syn_canary
        self.game_seed = game_seed
        self.include_auxilary = include_auxilary
        self.num_aux_in = num_aux_in
        self.combine_wt2_test = combine_wt2_test
        self.attach_id = attach_id


        #TODO: assert received split makes sense

        # Perform setup in init
        self.setup()

        # We don't setup the synthetic dataset in the constructor. The synthetic dataset might not be generated yet at the time of constructing this object

        # Setting up the auxilary dataset in init
        # if self.include_auxilary:
        #     self.setup_auxilary_dataset()
        

    def setup(self):
        """

        """
        # Handling the real dataset
        # Loading datasets dict. datasets_dict (datasets.DatasetDict) contains train, validation and test datasets.Dataset
        datasets_dict = load_dataset(self.path, self.name)

        if self.combine_wt2_test:
            combined_dataset = concatenate_datasets([datasets_dict['train'], datasets_dict['test']])

            datasets_dict['train'] = combined_dataset

            del datasets_dict['test']
        
        if self.include_auxilary:
            datasets_dict = load_dataset(
                'wikitext', 
                'wikitext-103-raw-v1', 
            )

            train_dataset = load_dataset(
                'wikitext', 
                'wikitext-103-raw-v1', 
                split='train[:4%]' # this 4% is hardcoded in the code. Should be added as an argument 
            )

            datasets_dict['train'] = train_dataset

            del datasets_dict['test']




        logging.info(f"Original datasets description:\n{datasets_dict}")

        logging.info(f"First 5 examples in the unshuffled original dataset:\n{datasets_dict['train'][:5]}")

        # Instantiating a tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, use_fast=True)
        
        # increasing the model max length. We would later break the dataset into chunks that are smaller than Model length
        self.tokenizer.model_max_length = sys.maxsize 

        # Lambda function for tokenizing the datasets
        

        # datasets.DatasetDict has a map method applying a function to all the elements in the table
        # The transformation is applied to all the datasets of the dataset dictionary.
        self.tokenize_lambda_function = lambda examples: self._tokenize_function(examples, self.tokenizer)
        tokenized_datasets_dict = datasets_dict.map(self.tokenize_lambda_function, batched=True, batch_size=1000, num_proc=4, remove_columns=["text"])

        # Preprocessing the dataset. Breaking the dataset into equal chunks
        group_texts_lambda_function = lambda examples: self._group_texts(examples, self.block_size)
        chunked_datasets_dict = tokenized_datasets_dict.map(
            group_texts_lambda_function,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )
        logging.info(f"Original chunked datasets description:\n{chunked_datasets_dict}")

        # Shuffle the dataset. This ensures inclusion/exclusion into the target model training set is definitely random.
        if self.do_shuffle:
            self.shuffled_chunked_datasets_dict = chunked_datasets_dict.shuffle(seed=self.seed)
        logging.info(f"First 5 examples in the shuffled original dataset:\n{self.tokenizer.batch_decode(self.shuffled_chunked_datasets_dict['train']['input_ids'][:5])}")
        
        
        

    def setup_synthetic_dataset(self):
        """
        """
        if (self.path_to_synthetic_data is not None):
            # Loading synthetic dataset into a pandas dataframe
            synthetic_df = self._read_split(self.path_to_synthetic_data)

            # renaming the name of columnn containing the generated texts to text
            synthetic_df.rename(columns={self.synthetic_text_column_name: 'text'}, inplace=True)
            
            # selecting the generated text column. Used double brackets to keep it as a DataFrame instead of a Series
            synthetic_df = synthetic_df[['text']]

            # Creating a datasets.Dataset from synthetic samples
            synthetic_dataset = Dataset.from_pandas(synthetic_df)

            logging.info(f"Synthetic dataset description:\n{synthetic_dataset}")

            logging.info(f"First 5 examples in the unshuffled synthetic dataset:\n{synthetic_dataset[:5]}")

            # Tokenizing the synthetic dataset
            tokenized_synthetic_dataset = synthetic_dataset.map(self.tokenize_lambda_function, batched=True, num_proc=4, remove_columns=["text"])

            # Each sample in my synthetic dataset is controlled to have length of (block_size) tokens.
            # TODO: Control for my general cases if this is not the case

            # Shuffle the dataset. This ensures inclusion/exclusion into the target model training set is definitely random.
            if self.do_shuffle:
                self.shuffled_synthetic_dataset = tokenized_synthetic_dataset.shuffle(seed=self.seed)

            # logging to sanity check that the shuffled version is consistent among different modules in PANORAMIA
            logging.info(f"First 5 examples in the shuffled synthetic dataset:\n{self.tokenizer.batch_decode(self.shuffled_synthetic_dataset['input_ids'][:5])}")

        else:
            logging.info(f"Synthetic dataset not provided.")

    # def setup_auxilary_dataset(self):
    #     """
    #     In some experiments on WT-2 dataset, we need more real samples from the WT dataset.
    #     We source it from WT-103 dataset (from its train split).
    #     """
    #     # hard-coded on the split we take from WT-103. The last 20% doesn't have overlap with WT-2
    #     wt_103_train_dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train[-2%:]')

    #     logging.info(f"Chosen WT-103 split  description:\n{wt_103_train_dataset}")

    #     # Tokenizing the dataset
    #     tokenized_aux_dataset = wt_103_train_dataset.map(self.tokenize_lambda_function, batched=True, num_proc=4, remove_columns=["text"])

    #     # Preprocessing the dataset. Breaking the dataset into equal chunks
    #     group_texts_lambda_function = lambda examples: self._group_texts(examples, self.block_size)
    #     chunked_aux_dataset = tokenized_aux_dataset.map(
    #         group_texts_lambda_function,
    #         batched=True,
    #         batch_size=1000,
    #         num_proc=4,
    #     )

    #     logging.info(f"Chunked auxilary dataset description:\n{chunked_aux_dataset}")

    #     # Shuffle the dataset. This ensures inclusion/exclusion into the target model training set is definitely random.
    #     if self.do_shuffle:
    #         self.shuffled_aux_dataset = chunked_aux_dataset.shuffle(seed=self.seed)

    #     # logging to sanity check that the shuffled version is consistent among different modules in PANORAMIA
    #     logging.info(f"First 5 examples in the shuffled auxilary dataset:\n{self.tokenizer.batch_decode(self.shuffled_aux_dataset['input_ids'][:5])}")



    @staticmethod
    def _tokenize_function(examples, tokenizer):
            return tokenizer(examples["text"])

    @staticmethod
    def _group_texts(examples, block_size):
        """
        Concatenate the whole dataset and chunk it into chunks of equal size 
        """

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result   

    @staticmethod
    def _read_split(file_path):
        df = pd.read_csv(file_path)
        return df
    
    @staticmethod
    def _convert_percent_to_index(percentage, length):
        return length * percentage // 100

    def get_generator_training_offset(self):
        return self._convert_percent_to_index(self.generator_train_percent, len(self.shuffled_chunked_datasets_dict['train']))

    def get_generator_prompting_offset(self):
        return self._convert_percent_to_index(self.generator_train_percent + self.prompt_sampling_percent, len(self.shuffled_chunked_datasets_dict['train']))

    def get_target_model_training_offset(self):
        return self._convert_percent_to_index(self.target_model_percent, len(self.shuffled_chunked_datasets_dict['train']))

    def get_helper_model_offset(self):
        if self.helper_model_train_data_mode == 'real':
            return self._convert_percent_to_index(self.target_model_percent + self.helper_model_percent, len(self.shuffled_chunked_datasets_dict['train']))
        elif self.helper_model_train_data_mode == 'syn':
            return self._convert_percent_to_index(self.target_model_percent, len(self.shuffled_chunked_datasets_dict['train']))
        else:
            raise NotImplementedError

    def get_syn_in_offset(self):
        # indices of synthetic samples included in the target model 0:self.num_syn_canary
        return self.num_syn_canary

    def _split_syn_audit_helper(self):
        syn_audit_offset = self._convert_percent_to_index(
            self.syn_audit_percent, 
            len(self.shuffled_synthetic_dataset)
        )
        syn_audit = self.shuffled_synthetic_dataset.select(range(syn_audit_offset))
        syn_helper = self.shuffled_synthetic_dataset.select(range(syn_audit_offset, len(self.shuffled_synthetic_dataset)))
        logging.info(f"Splitted the synthetic dataset into:\nSynthetic dedicated for audit:{syn_audit}\nSynthetic dedicated for training the helper model: {syn_helper}")
        return syn_audit, syn_helper

    def _split_syn_audit_dataset_in_out(self):
        syn_audit, _ = self._split_syn_audit_helper()
        syn_in_dataset = syn_audit.select(range(self.get_syn_in_offset()))
        syn_out_dataset = syn_audit.select(range(self.get_syn_in_offset(), len(syn_audit)))
        return syn_in_dataset, syn_out_dataset
    
    def _split_syn_helper_train_val(self):
        _, syn_helper = self._split_syn_audit_helper()

        # TODO: hardcoding the number of validation samples. Modify it later
        num_validation = 1800

        syn_train_offset = self._convert_percent_to_index(self.helper_model_percent, len(syn_helper)-1800)

        syn_helper_train = syn_helper.select(range(syn_train_offset))
        syn_helper_val = syn_helper.select(range(len(syn_helper) - num_validation, len(syn_helper)))
        syn_helper_test = None
        return syn_helper_train, syn_helper_val, syn_helper_test

    def get_real_member_audit_dataset(self):
        train_offset = self.get_target_model_training_offset()
        generator_prompting_offset = self.get_generator_prompting_offset()
        return self.shuffled_chunked_datasets_dict['train'].select(range(generator_prompting_offset, train_offset))

    def get_length_real_member_audit_dataset(self):
        return len(self.get_real_member_audit_dataset())

    def get_real_non_member_audit_dataset(self):
        helper_train_offset = self.get_helper_model_offset()
        RNM_from_train = self.shuffled_chunked_datasets_dict['train'].select(range(helper_train_offset, len(self.shuffled_chunked_datasets_dict['train'])))
        if self.combine_wt2_test or self.include_auxilary:
            RNM_audit_dataset = RNM_from_train
        else:
            RNM_from_test = self.shuffled_chunked_datasets_dict['test']
            RNM_audit_dataset = concatenate_datasets([RNM_from_train, RNM_from_test])
        logging.info(f"Real Non-member Audit dataset description:\n{RNM_audit_dataset}")
        return RNM_audit_dataset

    def get_length_real_non_member_audit_dataset(self):
        return len(self.get_real_non_member_audit_dataset())

    # def _split_aux_audit_dataset_in_out(self):
    #     aux_in = self.shuffled_aux_dataset.select(
    #         range(self.num_aux_in)
    #     )
    #     aux_out = self.shuffled_aux_dataset.select(
    #         range(self.num_aux_in, len(self.shuffled_aux_dataset))
    #     )
    #     logging.info(f"Splitted the auxilary dataset into:\nAuxilary dedicated for audit real members:{aux_in}\nAuxilary dedicated for audit real non-members: {aux_out}")
    #     return aux_in, aux_out

    def get_target_model_datasets(self):
        # last index of the allocated dataset to the target model
        train_offset = self.get_target_model_training_offset()
        val_offset = self._convert_percent_to_index(self.target_model_percent, len(self.shuffled_chunked_datasets_dict['validation']))

        # selecting the datasets 
        train_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(train_offset))
        val_dataset = self.shuffled_chunked_datasets_dict['validation'].select(range(val_offset))
        test_dataset = None

        if self.include_synthetic:
            logging.info(f"including {self.num_syn_canary} synthetic samples into the target model...")
            syn_in_dataset, syn_out_dataset = self._split_syn_audit_dataset_in_out()
            train_dataset = concatenate_datasets([train_dataset, syn_in_dataset])

        # if self.include_auxilary:
        #     logging.info(f"including {self.num_aux_in} auxilary samples into the target model...")
        #     aux_in_dataset, _ = self._split_aux_audit_dataset_in_out()
        #     train_dataset = concatenate_datasets([train_dataset, aux_in_dataset])

        # set the labels to input_ids (the task is language modeling)
        train_dataset = train_dataset.add_column('labels', train_dataset['input_ids'].copy())
        val_dataset = val_dataset.add_column('labels', val_dataset['input_ids'].copy())

        logging.info(f"ُTarget model train dataset:\n{train_dataset}")
        logging.info(f"Target model validation dataset:\n{val_dataset}")

        return train_dataset, val_dataset, test_dataset

    def get_target_model_dataloaders(self):
        raise NotImplementedError

    def get_helper_model_dataset(self):
        # TODO: assert helper model train data mode is consistent with helper model percent
        if self.helper_model_train_data_mode == 'real':

            # last index of the allocated dataset to the target model
            target_train_offset = self.get_target_model_training_offset()
            target_val_offset = self._convert_percent_to_index(self.target_model_percent, len(self.shuffled_chunked_datasets_dict['validation']))

            # last index of the allocated dataset to the helper model
            helper_train_offset = self.get_helper_model_offset()
            helper_val_offset = self._convert_percent_to_index(self.target_model_percent + self.helper_model_percent, len(self.shuffled_chunked_datasets_dict['validation']))

            # selecting the datasets 
            train_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(target_train_offset, helper_train_offset))
            val_dataset = self.shuffled_chunked_datasets_dict['validation'].select(range(target_val_offset, helper_val_offset))
            test_dataset = None
            
            

            logging.info(f"Helper model with real train dataset:\n{train_dataset}")
            logging.info(f"Helper model with real validation dataset:\n{val_dataset}")

            
        
        elif self.helper_model_train_data_mode == 'syn':
            train_dataset, val_dataset, test_dataset = self._split_syn_helper_train_val()

            logging.info(f"Helper model with synthetic train dataset:\n{train_dataset}")
            logging.info(f"Helper model with synthetic validation dataset:\n{val_dataset}")

            
        else:
            raise NotImplementedError
        
        # set the labels to input_ids (the task is language modeling)
        train_dataset = train_dataset.add_column('labels', train_dataset['input_ids'].copy())
        val_dataset = val_dataset.add_column('labels', val_dataset['input_ids'].copy())

        return train_dataset, val_dataset, test_dataset

    def get_helper_model_dataloaders(self):
        raise NotImplementedError

    def get_generator_training_datasets(self):
        # TODO: should the validation be the same split?
        
        # last index allocated to the generator training
        train_offset = self.get_generator_training_offset()
        val_offset = self._convert_percent_to_index(self.generator_train_percent, len(self.shuffled_chunked_datasets_dict['validation']))

        # selecting the allocated part of the dataset to the generator
        train_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(train_offset))
        val_dataset = self.shuffled_chunked_datasets_dict['validation'].select(range(val_offset))
        test_dataset = None

        # set the labels to input_ids (the task is language modeling)
        train_dataset = train_dataset.add_column('labels', train_dataset['input_ids'].copy())
        val_dataset = val_dataset.add_column('labels', val_dataset['input_ids'].copy())

        logging.info(f"Generator train dataset:\n{train_dataset}")
        logging.info(f"Generator validation dataset:\n{val_dataset}")
        return train_dataset, val_dataset, test_dataset

    def get_generator_training_dataloaders(self):
        raise NotImplementedError
    
    def get_generator_prompt_datasets(self):
        # TODO: provide functionalities to provide prompts from different modes: Real members, Real non-members (either from helper model train set, or original test set), fake-nonmembers
        
        # retrieving last index allocated for generator prompts
        generator_training_offset = self.get_generator_training_offset()
        generator_prompting_offset = self.get_generator_prompting_offset()

        #selecting the prompting section of the dataset
        prompt_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(generator_training_offset, generator_prompting_offset))

        logging.info(f"Generation prompt dataset:\n{prompt_dataset}")
        return prompt_dataset.with_format("torch")
        

    def get_generator_prompt_dataloaders(self, batch_size):
        torch_prompt_dataset = self.get_generator_prompt_datasets()
        return DataLoader(torch_prompt_dataset, batch_size=batch_size, shuffle=True)
        

    def _check_split_limit(self):
        ...
    
    def get_mia_datasets(self):
        generator_prompting_offset = self.get_generator_prompting_offset()
        target_train_offset = self.get_target_model_training_offset()
        if self.audit_mode == 'RMFN':
            if self.include_synthetic == True:
                raise NotImplementedError
            else:
                # choosing members (+1 label) for train, val, test 
                real_member_audit_dataset_length = self.get_length_real_member_audit_dataset()

                # ensure there are enough real members for audit
                # if self.include_auxilary:
                #     assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length + self.num_aux_in, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}. Total number of auxilary data: {self.num_aux_in}"
                # else:
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}"
                
                RM_train = self.get_real_member_audit_dataset().select(
                    range(self.mia_num_train) 
                )
                # if self.include_auxilary: 
                #     RM_val = self.get_real_member_audit_dataset().select(
                #         range(real_member_audit_dataset_length  - (self.mia_num_val), real_member_audit_dataset_length)
                #     )
                    
                #     aux_in_dataset, _ = self._split_aux_audit_dataset_in_out()
                #     RM_test = aux_in_dataset.select(
                #         range(self.mia_num_test)
                #     )
                # else:
                RM_val = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_member_audit_dataset_length - (self.mia_num_test))
                )
                RM_test = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test), real_member_audit_dataset_length)
                )
                
                # setting the labels
                RM_train = RM_train.add_column("labels", [1.]*len(RM_train))
                RM_val = RM_val.add_column("labels", [1.]*len(RM_val))

                syn_audit, _ = self._split_syn_audit_helper()

                # ensure there are enough synthetic non-members for audit
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= len(syn_audit), f"There are not enough audit synthetic non-member samples for spliting into train/val/test. Total number of audit synthetic non-members: {len(syn_audit)}"

                FN_train = syn_audit.select(
                    range(self.mia_num_train) 
                )

                FN_val = syn_audit.select(
                    range(len(syn_audit) - (self.mia_num_test) - (self.mia_num_val), len(syn_audit) - (self.mia_num_test))
                )

                FN_test = syn_audit.select(
                    range(len(syn_audit) - (self.mia_num_test), len(syn_audit))
                )

                # setting the labels
                FN_train = FN_train.add_column("labels", [0.]*len(FN_train))
                FN_val = FN_val.add_column("labels", [0.]*len(FN_val))

                # merging
                train = concatenate_datasets([RM_train, FN_train])
                val = concatenate_datasets([RM_val, FN_val])

                # use PANORAMIA game for the test set
                test = self.PANORAMIA_auditing_game(
                    x_in=RM_test,
                    x_gen=FN_test,
                    m=self.mia_num_test
                )

                logging.info(f"Providing the audit datasets in RMFN mode with the following details:")
                logging.info(f"Train dataset:\n{train}")
                logging.info(f"Validation dataset:\n{val}")
                logging.info(f"Test dataset:\n{test}")

                logging.info(f"First 5 examples in the Test (audit) dataset:\n{test[:5]}")

                return train.with_format("torch"), val.with_format("torch"), test.with_format("torch")
                    
        elif self.audit_mode == 'RMRN':
            if self.include_synthetic == True:
                raise NotImplementedError
            else:
                # choosing members (+1 label) for train, val, test 
                real_member_audit_dataset_length = self.get_length_real_member_audit_dataset()

                # ensure there are enough real members for audit
                # if self.include_auxilary:
                #     assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length + self.num_aux_in, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}. Total number of auxilary member data: {self.num_aux_in}"
                # else:
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}"
                
                RM_train = self.get_real_member_audit_dataset().select(
                    range(self.mia_num_train) 
                )
                # if self.include_auxilary: 
                #     RM_val = self.get_real_member_audit_dataset().select(
                #         range(real_member_audit_dataset_length  - (self.mia_num_val), real_member_audit_dataset_length)
                #     )
                    
                #     aux_in_dataset, _ = self._split_aux_audit_dataset_in_out()
                #     RM_test = aux_in_dataset.select(
                #         range(self.mia_num_test)
                #     )
                # else:
                RM_val = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_member_audit_dataset_length - (self.mia_num_test))
                )
                RM_test = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test), real_member_audit_dataset_length)
                )
                
                # setting the labels
                RM_train = RM_train.add_column("labels", [1.]*len(RM_train))
                RM_val = RM_val.add_column("labels", [1.]*len(RM_val))

                # Total real non-member samples available for audit
                RNM_audit = self.get_real_non_member_audit_dataset()

                # ensure there are enough real non-members for audit
                # if self.include_auxilary:
                #     assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= len(RNM_audit) + (len(self.shuffled_aux_dataset) - self.num_aux_in), f"There are not enough audit real non-member samples for spliting into train/val/test. Total number of audit real non-members: {len(RNM_audit)}. Total number of real auxilary non-member data:{(len(self.shuffled_aux_dataset) - self.num_aux_in)}"
                # else:
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= len(RNM_audit), f"There are not enough audit real non-member samples for spliting into train/val/test. Total number of audit real non-members: {len(RNM_audit)}"

                RN_train = RNM_audit.select(
                    range(self.mia_num_train) 
                )

                # if self.include_auxilary:
                #     RN_val = RNM_audit.select(
                #         range(len(RNM_audit)  - (self.mia_num_val), len(RNM_audit))
                #     )
                #     _, aux_out_dataset = self._split_aux_audit_dataset_in_out()
                #     RN_test = aux_out_dataset.select(
                #         range(self.mia_num_test)
                #     )
                # else:
                RN_val = RNM_audit.select(
                    range(len(RNM_audit) - (self.mia_num_test) - (self.mia_num_val), len(RNM_audit) - (self.mia_num_test))
                )

                RN_test = RNM_audit.select(
                    range(len(RNM_audit) - (self.mia_num_test), len(RNM_audit))
                )

                # setting the labels
                RN_train = RN_train.add_column("labels", [0.]*len(RN_train))
                RN_val = RN_val.add_column("labels", [0.]*len(RN_val))

                # merging
                train = concatenate_datasets([RM_train, RN_train])
                val = concatenate_datasets([RM_val, RN_val])

                # use PANORAMIA game for the test set
                test = self.PANORAMIA_auditing_game(
                    x_in=RM_test,
                    x_gen=RN_test,
                    m=self.mia_num_test
                )

                logging.info(f"Providing the audit datasets in RMRN mode with the following details:")
                logging.info(f"Train dataset:\n{train}")
                logging.info(f"Validation dataset:\n{val}")
                logging.info(f"Test dataset:\n{test}")

                logging.info(f"First 5 examples in the Test (audit) dataset:\n{test[:5]}")

                return train.with_format("torch"), val.with_format("torch"), test.with_format("torch")


        elif self.audit_mode == "RMRN_shuffle":
            if self.include_synthetic == True:
                raise NotImplementedError
            else:
                # choosing members (+1 label) for train, val, test 
                real_member_audit_dataset_length = self.get_length_real_member_audit_dataset()

                # ensure there are enough real members for audit
                # if self.include_auxilary:
                #     assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length + self.num_aux_in, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}. Total number of auxilary member data: {self.num_aux_in}"
                # else:
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= real_member_audit_dataset_length, f"There are not enough audit real member samples for spliting into train/val/test. Total number of audit real members: {real_member_audit_dataset_length}"
                
                RM_train = self.get_real_member_audit_dataset().select(
                    range(self.mia_num_train) 
                )
                # if self.include_auxilary: 
                #     RM_val = self.get_real_member_audit_dataset().select(
                #         range(real_member_audit_dataset_length  - (self.mia_num_val), real_member_audit_dataset_length)
                #     )
                    
                #     aux_in_dataset, _ = self._split_aux_audit_dataset_in_out()
                #     RM_test = aux_in_dataset.select(
                #         range(self.mia_num_test)
                #     )
                # else:
                RM_val = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_member_audit_dataset_length - (self.mia_num_test))
                )
                RM_test = self.get_real_member_audit_dataset().select(
                    range(real_member_audit_dataset_length - (self.mia_num_test), real_member_audit_dataset_length)
                )
                
                # shuffle randomly
                RM = concatenate_datasets([RM_train, RM_val, RM_test]).shuffle(self.game_seed)
                RM_train = RM.select(
                    range(self.mia_num_train)
                )
                RM_val = RM.select(
                    range(self.mia_num_train, self.mia_num_train+ self.mia_num_val)
                )
                RM_test = RM.select(
                    range(self.mia_num_train + self.mia_num_val, self.mia_num_train + self.mia_num_val + self.mia_num_test)
                )

                # setting the labels
                RM_train = RM_train.add_column("labels", [1.]*len(RM_train))
                RM_val = RM_val.add_column("labels", [1.]*len(RM_val))

                # Total real non-member samples available for audit
                RNM_audit = self.get_real_non_member_audit_dataset()

                # ensure there are enough real non-members for audit
                # if self.include_auxilary:
                #     assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= len(RNM_audit) + (len(self.shuffled_aux_dataset) - self.num_aux_in), f"There are not enough audit real non-member samples for spliting into train/val/test. Total number of audit real non-members: {len(RNM_audit)}. Total number of real auxilary non-member data:{(len(self.shuffled_aux_dataset) - self.num_aux_in)}"
                # else:
                assert self.mia_num_train + self.mia_num_val + self.mia_num_test <= len(RNM_audit), f"There are not enough audit real non-member samples for spliting into train/val/test. Total number of audit real non-members: {len(RNM_audit)}"

                RN_train = RNM_audit.select(
                    range(self.mia_num_train) 
                )

                # if self.include_auxilary:
                #     RN_val = RNM_audit.select(
                #         range(len(RNM_audit)  - (self.mia_num_val), len(RNM_audit))
                #     )
                #     _, aux_out_dataset = self._split_aux_audit_dataset_in_out()
                #     RN_test = aux_out_dataset.select(
                #         range(self.mia_num_test)
                #     )
                # else:
                RN_val = RNM_audit.select(
                    range(len(RNM_audit) - (self.mia_num_test) - (self.mia_num_val), len(RNM_audit) - (self.mia_num_test))
                )

                RN_test = RNM_audit.select(
                    range(len(RNM_audit) - (self.mia_num_test), len(RNM_audit))
                )

                RN = concatenate_datasets([RN_train, RN_val, RN_test]).shuffle(self.game_seed)
                RN_train = RN.select(
                    range(self.mia_num_train)
                )
                RN_val = RN.select(
                    range(self.mia_num_train, self.mia_num_train + self.mia_num_val)
                )
                RN_test = RN.select(
                    range(self.mia_num_train + self.mia_num_val, self.mia_num_train + self.mia_num_val + self.mia_num_test)
                )

                # setting the labels
                RN_train = RN_train.add_column("labels", [0.]*len(RN_train))
                RN_val = RN_val.add_column("labels", [0.]*len(RN_val))

                # merging
                train = concatenate_datasets([RM_train, RN_train])
                val = concatenate_datasets([RM_val, RN_val])

                # use PANORAMIA game for the test set
                test = self.PANORAMIA_auditing_game(
                    x_in=RM_test,
                    x_gen=RN_test,
                    m=self.mia_num_test
                )

                logging.info(f"Providing the audit datasets in RMRN mode with the following details:")
                logging.info(f"Train dataset:\n{train}")
                logging.info(f"Validation dataset:\n{val}")
                logging.info(f"Test dataset:\n{test}")

                logging.info(f"First 5 examples in the Test (audit) dataset:\n{test[:5]}")

                return train.with_format("torch"), val.with_format("torch"), test.with_format("torch")
        elif self.audit_mode == 'RMFMRNFN':
            assert self.include_synthetic == True, "The target model needs to have synthetic samples inside the target trainin set"
            
            # choosing members (+1 label) for train, val, test 
            real_member_audit_dataset_length = self.get_length_real_member_audit_dataset()
            
            RM_train = self.get_real_member_audit_dataset().select(
                range(self.mia_num_train // 2) # mia_num_train is per class
            )
            RM_val = self.get_real_member_audit_dataset().select(
                range(real_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_member_audit_dataset_length - (self.mia_num_test))
            )
            RM_val = RM_val.select(
                range(self.mia_num_val//2)
            )
            RM_test = self.get_real_member_audit_dataset().select(
                range(real_member_audit_dataset_length - (self.mia_num_test), real_member_audit_dataset_length)
            )
            RM_test = RM_test.select(
                range(self.mia_num_test//2)
            )

            syn_in_dataset, syn_out_dataset = self._split_syn_audit_dataset_in_out()

            FM_train = syn_in_dataset.select(
                range(self.mia_num_train // 2)
            )

            FM_val = syn_in_dataset.select(
                range(self.num_syn_canary - (self.mia_num_test//2) - (self.mia_num_val//2), self.num_syn_canary - (self.mia_num_test//2))
            )
            
            FM_test = syn_in_dataset.select(
                range(self.num_syn_canary - (self.mia_num_test//2), self.num_syn_canary)
            )

            # setting the labels
            M_train = concatenate_datasets([RM_train, FM_train])
            M_val = concatenate_datasets([RM_val, FM_val])

            M_train = M_train.add_column("labels", [1.]*len(M_train))
            M_val = M_val.add_column("labels", [1.]*len(M_val))
            

            # choosing non-members (0 label) for train, val, test 
            real_non_member_audit_dataset_length = self.get_length_real_non_member_audit_dataset()

            RN_train = self.get_real_non_member_audit_dataset().select(
                range(self.mia_num_train // 2) # mia_num_train is per class
            )
            RN_val = self.get_real_non_member_audit_dataset().select(
                range(real_non_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_non_member_audit_dataset_length - (self.mia_num_test))
            )
            RN_val = RN_val.select(
                range(self.mia_num_val//2)
            )
            RN_test = self.get_real_non_member_audit_dataset().select(
                range(real_non_member_audit_dataset_length - (self.mia_num_test), real_non_member_audit_dataset_length)
            )
            RN_test = RN_test.select(
                range(self.mia_num_test//2)
            )

            syn_out_length = len(syn_out_dataset)
            FN_train = syn_out_dataset.select(
                range(self.mia_num_train // 2)
            )
            FN_val = syn_out_dataset.select(
                range(syn_out_length - (self.mia_num_test) - (self.mia_num_val), syn_out_length - (self.mia_num_test))
            )
            FN_val = FN_val.select(
                range(self.mia_num_val//2)
            )
            FN_test = syn_out_dataset.select(
                range(syn_out_length - (self.mia_num_test), syn_out_length)
            )
            FN_test = FN_test.select(
                range(self.mia_num_test//2)
            )

            # setting the labels

            N_train = concatenate_datasets([RN_train, FN_train])
            N_val = concatenate_datasets([RN_val, FN_val])

            N_train = N_train.add_column("labels", [0.]*len(N_train))
            N_val = N_val.add_column("labels", [0.]*len(N_val))

            # merging
            train = concatenate_datasets([M_train, N_train])
            val = concatenate_datasets([M_val, N_val])

            if self.attach_id:
                RM_test = RM_test.add_column("type", [1.]*len(RM_test))
                FM_test = FM_test.add_column("type", [2.]*len(FM_test))
                RN_test = RN_test.add_column("type", [3.]*len(RN_test))
                FN_test = FN_test.add_column("type", [4.]*len(FN_test))
                
            
            
            # use PANORAMIA game for the test set
            test = self.PANORAMIA_auditing_game(
                x_in=concatenate_datasets([RM_test, FM_test]),
                x_gen=concatenate_datasets([RN_test, FN_test]),
                m=self.mia_num_test
            )
            print(test)

            logging.info(f"Providing the audit datasets in RMFMRNFN mode with the following details:")
            logging.info(f"Train dataset:\n{train}")
            logging.info(f"Validation dataset:\n{val}")
            logging.info(f"Test dataset:\n{test}")

            return train.with_format("torch"), val.with_format("torch"), test.with_format("torch")


        elif self.audit_mode == 'FMFN':
            raise NotImplementedError
        elif self.audit_mode == 'RMFMFN':
            assert self.include_synthetic == True, "The target model needs to have synthetic samples inside the target trainin set"
            
            # choosing members (+1 label) for train, val, test 
            real_member_audit_dataset_length = self.get_length_real_member_audit_dataset()
            
            RM_train = self.get_real_member_audit_dataset().select(
                range(self.mia_num_train // 2) # mia_num_train is per class
            )
            RM_val = self.get_real_member_audit_dataset().select(
                range(real_member_audit_dataset_length - (self.mia_num_test) - (self.mia_num_val), real_member_audit_dataset_length - (self.mia_num_test))
            )
            RM_val = RM_val.select(
                range(self.mia_num_val//2)
            )
            RM_test = self.get_real_member_audit_dataset().select(
                range(real_member_audit_dataset_length - (self.mia_num_test), real_member_audit_dataset_length)
            )
            RM_test = RM_test.select(
                range(self.mia_num_test//2)
            )

            syn_in_dataset, syn_out_dataset = self._split_syn_audit_dataset_in_out()

            FM_train = syn_in_dataset.select(
                range(self.mia_num_train // 2)
            )
            FM_val = syn_in_dataset.select(
                range(self.num_syn_canary - (self.mia_num_test//2) - (self.mia_num_val//2), self.num_syn_canary - (self.mia_num_test//2))
            )
            FM_test = syn_in_dataset.select(
                range(self.num_syn_canary - (self.mia_num_test//2), self.num_syn_canary)
            )

            # setting the labels
            M_train = concatenate_datasets([RM_train, FM_train])
            M_val = concatenate_datasets([RM_val, FM_val])

            M_train = M_train.add_column("labels", [1.]*len(M_train))
            M_val = M_val.add_column("labels", [1.]*len(M_val))
            

            # choosing non-members (0 label) for train, val, test 

            syn_out_length = len(syn_out_dataset)
            FN_train = syn_out_dataset.select(
                range(self.mia_num_train)
            )
            FN_val = syn_out_dataset.select(
                range(syn_out_length - (self.mia_num_test) - (self.mia_num_val), syn_out_length - (self.mia_num_test))
            )
            FN_test = syn_out_dataset.select(
                range(syn_out_length - (self.mia_num_test), syn_out_length)
            )

            # setting the labels

            N_train = FN_train
            N_val = FN_val

            N_train = N_train.add_column("labels", [0.]*len(N_train))
            N_val = N_val.add_column("labels", [0.]*len(N_val))

            # merging
            train = concatenate_datasets([M_train, N_train])
            val = concatenate_datasets([M_val, N_val])

            # use PANORAMIA game for the test set
            test = self.PANORAMIA_auditing_game(
                x_in=concatenate_datasets([RM_test, FM_test]),
                x_gen=FN_test,
                m=self.mia_num_test
            )

            logging.info(f"Providing the audit datasets in RMFMFN mode with the following details:")
            logging.info(f"Train dataset:\n{train}")
            logging.info(f"Validation dataset:\n{val}")
            logging.info(f"Test dataset:\n{test}")

            return train.with_format("torch"), val.with_format("torch"), test.with_format("torch")
        else:
            raise NotImplementedError

    def get_mia_dataloaders(self):
        raise NotImplementedError

    def _get_random_bit_sequence(self, m: int):
        rng = np.random.default_rng(self.game_seed)
        return rng.integers(low=0, high=2, size=m).astype("float").tolist()

    def PANORAMIA_auditing_game(self, 
    x_in: Dataset, 
    x_gen: Dataset,
    m: int
    ):
        """
        
        """
        assert len(x_in) == m, "x_in doesn't have m samples"
        assert len(x_gen) == m, "x_gen doesn't have m samples"
        # TODO: assert datasets has no column called labels
        

        # m random binary bits
        s = self._get_random_bit_sequence(m)
        
        # adding s to the datasets to filter them based on the bits
        x_in = x_in.add_column("labels", s)
        x_gen = x_gen.add_column("labels", s)

        # filter samples based on the coin flips
        picked_x_in = x_in.filter(lambda example: example['labels'] == 1.)
        picked_x_gen = x_gen.filter(lambda example: example['labels'] == 0.)

        # attaching the chosen samples from each pair together
        x = concatenate_datasets([picked_x_in, picked_x_gen])

        # shuffle dataset 
        # TODO: Reproducibility?
        # x = x.shuffle()

        logging.info(f"Auditing set:\n{x}")

        # TODO: should it be returned in torch format? 
        return x
