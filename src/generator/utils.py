import logging

import pandas as pd
import numpy as np

def check_length_to_block_size(synthetic_data: pd.DataFrame, tokenizer, block_size):
    encodings = tokenizer(synthetic_data['text'].values.tolist())

    lengths = np.array([len(encodings["input_ids"][i]) for i in range(len(synthetic_data))])

    logging.info(f"Number of synthetic samples with length=block_size={block_size}")
    logging.info(np.sum(lengths == np.array([block_size for i in range(len(synthetic_data))])))
    
    neq_block_size_indices = np.where(lengths != np.array([block_size for i in range(len(synthetic_data))]))[0]

    for idx in neq_block_size_indices[::-1]:
        synthetic_data.drop(idx, inplace=True)
    
    return synthetic_data