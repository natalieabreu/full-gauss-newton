# pretokenize_dataset.py

import argparse
import os
import json
import base64
import numpy as np
from functools import partial
from multiprocessing import Pool
from datasets import load_dataset, Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer
import mlxu  # Ensure mlxu is installed and accessible


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field.startswith('<|') and field.endswith('|>'):
                # Special tokens.
                field = field[2:-2]
                if field == 'bos':
                    token_buffer.append(self.tokenizer.bos_token_id)
                elif field == 'eos':
                    token_buffer.append(self.tokenizer.eos_token_id)
                else:
                    # Token ID specified directly.
                    token_buffer.append(int(field))
                loss_mask_buffer.append(mask)
            elif field.startswith('{') and field.endswith('}'):
                field = field[1:-1]
                # Base64 encoded raw tokens.
                tokens = np.frombuffer(
                    base64.b64decode(example[field]),
                    dtype=self.config.base64_token_dtype
                ).tolist()
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return {
            'input_tokens': token_buffer,
            'loss_masks': loss_mask_buffer,
            **({} if not has_aux else {'aux': aux})
        }

def pretokenize_dataset(
    dataset_path,
    dataset_name,
    split,
    tokenizer_name,
    text_processor_config,
    output_dir,
    num_proc=4,
    batch_size=32,
    save_after_batches=1000
):
    """
    Pre-tokenizes the dataset and saves it to disk.

    Args:
        dataset_path (str): The Huggingface dataset path.
        dataset_name (str): The specific dataset name (can be None).
        split (str): The split to tokenize (e.g., 'train').
        tokenizer_name (str): The Huggingface tokenizer name.
        text_processor_config (dict): Configuration for TextProcessor.
        output_dir (str): Directory to save the tokenized dataset.
        num_proc (int): Number of processes for parallel tokenization.
        batch_size (int): Number of examples per batch.
        save_after_batches (int): Number of batches after which to save intermediate results.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Initialize TextProcessor
    text_processor = TextProcessor(text_processor_config, tokenizer)

    # Load dataset
    print(f"Loading dataset: {dataset_path}, name: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_path, name=dataset_name, split=split, streaming=False)

    # Define the processing function
    def process_example(example):
        return text_processor(example)

    # Apply the processing in parallel
    print("Starting tokenization...")
    tokenized_dataset = dataset.map(
        process_example,
        batched=False,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Define the new features for the tokenized dataset
    features = Features({
        'input_tokens': Sequence(Value('int32')),
        'loss_masks': Sequence(Value('float32')),
        # Include any auxiliary fields if present
        # 'aux': ...
    })

    # Convert to DatasetDict if necessary
    if isinstance(tokenized_dataset, Dataset):
        tokenized_dataset = DatasetDict({'train': tokenized_dataset})

    # Save the tokenized dataset to disk
    print(f"Saving tokenized dataset to {output_dir}...")
    tokenized_dataset.save_to_disk(output_dir)
    print("Pre-tokenization and saving completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pre-tokenization Script with Specified Arguments.")

    # Define arguments
    parser.add_argument('--tokenizer', type=str, required=True, help='Name of the Huggingface tokenizer (e.g., "google-t5/t5-base").')
    parser.add_argument('--train_dataset_type', type=str, default='huggingface', help='Type of the training dataset.')
    parser.add_argument('--train_dataset_text_processor_fields', type=str, required=True, help='Fields to process in the TextProcessor.')
    parser.add_argument('--train_dataset_text_processor_add_bos_token', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to add BOS token.')
    parser.add_argument('--train_dataset_huggingface_dataset_path', type=str, required=True, help='Path of the Huggingface dataset (e.g., "allenai/c4").')
    parser.add_argument('--train_dataset_huggingface_dataset_name', type=str, default=None, help='Name of the dataset (can be None).')
    parser.add_argument('--train_dataset_huggingface_dataset_streaming', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use streaming.')
    parser.add_argument('--train_dataset_huggingface_dataset_split', type=str, default='train', help='Dataset split to tokenize (e.g., "train").')
    parser.add_argument('--train_dataset_batch_size', type=int, default=32, help='Batch size for tokenization.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the tokenized dataset.')
    parser.add_argument('--num_proc', type=int, default=4, help='Number of processes for parallel tokenization.')

    args = parser.parse_args()

    print(args)

    # Verify that train_dataset_type is 'huggingface'
    if args.train_dataset_type != 'huggingface':
        raise ValueError("Currently, only 'huggingface' dataset type is supported.")

    # Build the text_processor_config
    text_processor_config = {
        'fields_from_example': '',  # As per the user's argument, not specified
        'fields': args.train_dataset_text_processor_fields,
        'subfield_separator': ' ',
        'add_bos_token': args.train_dataset_text_processor_add_bos_token,
        'add_eos_token': True,  # Assuming True, adjust if needed
        'prepend_text': '',
        'base64_token_dtype': 'i4',
    }

    pretokenize_dataset(
        dataset_path=args.train_dataset_huggingface_dataset_path,
        dataset_name=args.train_dataset_huggingface_dataset_name,
        split=args.train_dataset_huggingface_dataset_split,
        tokenizer_name=args.tokenizer,
        text_processor_config=text_processor_config,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        batch_size=args.train_dataset_batch_size,
    )
