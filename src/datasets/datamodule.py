import logging
import os
from typing import Optional
import random

import torch
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
        path_to_synthetic_data_dir: str = None,
        name_synthetic_data: str = None,
        synthetic_text_column_name: str = 'text',
        seed:  Optional[int] = None,
        do_shuffle: bool = False,
        pretrained_model_name_or_path: str = 'gpt2',
        block_size: int = 128,
        generator_train_percent: int = 25,
        prompt_sampling_percent: int = 13,
        target_model_percent: int = 50,
        helper_model_percent: int = 50,
        mia_real_num_train : int = 500,
        mia_real_num_val : int = 500,
        mia_real_num_test : int = 1000,
        include_synthetic: bool = False,
        audit_mode: str = '',
        num_syn_canary: int = 2000
        
    ) -> None:
        """
        Parameters
        ----------
        helper_model_percent: int
            allocates the last (helper_model_percent)% of the original dataset for training of the helper model.
        """
        self.path = path
        self.name = name
        self.path_to_synthetic_data_dir = path_to_synthetic_data_dir
        self.name_synthetic_data = name_synthetic_data
        self.synthetic_text_column_name = synthetic_text_column_name
        self.seed = seed,
        self.do_shuffle = do_shuffle
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.block_size = block_size
        self.generator_train_percent = generator_train_percent
        self.prompt_sampling_percent = prompt_sampling_percent
        self.target_model_percent = target_model_percent
        self.helper_model_percent = helper_model_percent
        self.mia_real_num_train = mia_real_num_train
        self.mia_real_num_val = mia_real_num_val
        self.mia_real_num_test = mia_real_num_test # m in theory
        self.include_synthetic = include_synthetic
        self.audit_mode = audit_mode
        self.num_syn_canary = num_syn_canary

        #TODO: assert received split makes sense

        # Perform setup in init
        self.setup()

            

    def setup(self):
        """

        """
        # Handling the real dataset
        # Loading datasets dict. datasets_dict (datasets.DatasetDict) contains train, validation and test datasets.Dataset
        datasets_dict = load_dataset(self.path, self.name)

        logging.info(f"Original datasets description:\n{datasets_dict}")

        logging.info(f"First 5 examples in the unshuffled original dataset:\n{datasets_dict['train'][:5]}")

        # Instantiating a tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path, use_fast=True)

        # Lambda function for tokenizing the datasets
        

        # datasets.DatasetDict has a map method applying a function to all the elements in the table
        # The transformation is applied to all the datasets of the dataset dictionary.
        self.tokenize_lambda_function = lambda examples: self._tokenize_function(examples, self.tokenizer)
        tokenized_datasets_dict = datasets_dict.map(self.tokenize_lambda_function, batched=True, num_proc=4, remove_columns=["text"])

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
        
        # Handling the synthetic dataset
        self.setup_synthetic_dataset()
        
        

    def setup_synthetic_dataset(self):
        """
        """
        if (self.path_to_synthetic_data_dir is not None) and (self.name_synthetic_data is not None):
            # Loading synthetic dataset into a pandas dataframe
            synthetic_data_path = os.path.join(self.path_to_synthetic_data_dir, self.name_synthetic_data)
            synthetic_df = self._read_split(synthetic_data_path)

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

            # Each sample in my synthetic dataset is controlled to have length of 128 tokens.
            # TODO: Control for my general cases if this is not the case

            # Shuffle the dataset. This ensures inclusion/exclusion into the target model training set is definitely random.
            if self.do_shuffle:
                self.shuffled_synthetic_dataset = tokenized_synthetic_dataset.shuffle(seed=self.seed)

            # logging to sanity check that the shuffled version is consistent among different modules in PANORAMIA
            logging.info(f"First 5 examples in the shuffled synthetic dataset:\n{self.tokenizer.batch_decode(self.shuffled_synthetic_dataset['input_ids'][:5])}")

        else:
            logging.info(f"Synthetic dataset not provided.")

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

    def get_helper_model_first_index(self):
        return self._convert_percent_to_index((100 - self.helper_model_percent), len(self.shuffled_chunked_datasets_dict['train']))

    def get_target_model_datasets(self):
        # last index of the allocated dataset to the target model
        train_offset = self.get_target_model_training_offset()
        val_offset = self._convert_percent_to_index(self.target_model_percent, len(self.shuffled_chunked_datasets_dict['validation']))

        # selecting the datasets 
        train_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(train_offset))
        val_dataset = self.shuffled_chunked_datasets_dict['validation'].select(range(val_offset))
        test_dataset = None

        
        if include_synthetic:
            pass
        else:
            pass

        pass

    def get_target_model_dataloaders(self):
        pass

    def get_helper_model_dataset(self):
        pass

    def get_helper_model_dataloaders(self):
        pass

    def get_generator_training_datasets(self):
        # TODO: should the validation be the same split?
        
        # last index allocated to the generator training
        train_offset = self.get_generator_training_offset()
        val_offset = self._convert_percent_to_index(self.generator_train_percent, len(self.shuffled_chunked_datasets_dict['validation']))

        # selecting the allocated part of the dataset to the generator
        train_dataset = self.shuffled_chunked_datasets_dict['train'].select(range(train_offset))
        val_dataset = self.shuffled_chunked_datasets_dict['validation'].select(range(val_offset))
        test_dataset = None

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
        return prompt_dataset
        

    def get_generator_prompt_dataloaders(self):
        raise NotImplementedError

    def get_mia_datasets(self):
        pass

    def get_mia_dataloaders(self):
        pass

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
        s = np.random.randint(0, 2, size=m).tolist()
        
        # adding s to the datasets to filter them based on the bits
        x_in = x_in.add_column("labels", s)
        x_gen = x_gen.add_column("labels", s)

        # filter samples based on the coin flips
        picked_x_in = x_in.filter(lambda example: example['labels'] == 1)
        picked_x_gen = x_gen.filter(lambda example: example['labels'] == 0)

        # attaching the chosen samples from each pair together
        x = concatenate_datasets([picked_x_in, picked_x_gen])

        # shuffle dataset 
        x = x.shuffle()

        logging.info(f"Auditing set:\n{x}")

        # TODO: should it be returned in torch format? 
        return x.shuffle()
