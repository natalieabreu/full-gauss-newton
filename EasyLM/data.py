import time
from functools import partial
import json
import base64
from multiprocessing import Pool

import mlxu
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset

import jax


import os

class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = OptHuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface' and config.huggingface_dataset.pretokenized_dataset_dir != '':
            return OptHuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'huggingface':
            # raise ValueError('Huggingface dataset is not supported in this version.')
            return HuggingfaceDataset(config.huggingface_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


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

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = 'allenai/c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'
        config.tokens_count_at_start = 0        

        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        print(config,self.config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )
        self.metadata = {
            'dataset_example_index': 0,
            'dataset_total_tokens': self.config.tokens_count_at_start,
        }

        # if self.config.tokens_count_at_start > 0:
        #     print(f"Skipping {self.config.tokens_count_at_start} tokens at the start of the dataset.", flush=True)
        #     steps_to_skip = self.config.tokens_count_at_start // (self.config.seq_length * self.config.batch_size)
        #     self._dataset = self._dataset.skip(steps_to_skip)
        #     self.start_steps = steps_to_skip
        #     print(f"Skipped {steps_to_skip} steps.", flush=True)


    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = self.metadata['dataset_total_tokens']

        tokens_to_skip = self.config.tokens_count_at_start
        # tokens_to_skip = 0

        # Iterator over the dataset
        dataset_iterator = iter(self._dataset)
        

        # # NA: Added this to skip tokens at the start for resuming mid training
        # while tokens_to_skip > 0:
        #     skip_token_buffer = []
        #     skip_loss_mask_buffer = []
        #     try:
        #         example = next(dataset_iterator)
        #     except StopIteration:
        #         # End of dataset reached before skipping desired tokens
        #         print("Reached end of dataset while skipping tokens.")
        #         return
            
        #     start_steps = 0
        #     while len(skip_token_buffer) < chunk_size + 1:
        #         tokens, loss_masks = self.text_processor(example)
        #         skip_token_buffer.extend(tokens)
        #         skip_loss_mask_buffer.extend(loss_masks)

            # while len(skip_token_buffer) > chunk_size + 1:
            #     if tokens_to_skip <= 0:
            #         break
            
            #     tokens = np.array(skip_token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
            #                     self.config.batch_size, -1
            #                 )
            #     loss_masks = np.array(skip_loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
            #                 self.config.batch_size, -1
            #             ),
            #     skip_token_buffer = skip_token_buffer[chunk_size:]
            #     skip_loss_mask_buffer = skip_loss_mask_buffer[chunk_size:]

            #     tokens_to_skip -= chunk_size
            #     start_steps += 1
      
        # print(f"Created iterator; Starting from step {start_steps}.", flush=True)
        while True:
            token_buffer = []
            loss_mask_buffer = []

            # if len(skip_token_buffer) != 0: # NA: Resuming mid training
            #     token_buffer.extend(skip_token_buffer)
            #     loss_mask_buffer.extend(skip_loss_mask_buffer)
            #     skip_token_buffer = []
            #     skip_loss_mask_buffer = []

            for index, example in enumerate(dataset_iterator, start=self.metadata['dataset_example_index']):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    self.metadata = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, self.metadata
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        print(f"Saving state; Starting from step {self.metadata['dataset_example_index']}.", flush=True)
        self.metadata = jax.device_get(self.metadata)
        return dict(config=self.config, metadata=self.metadata)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))
        self.metadata = state_dict.get('metadata', self.metadata)
        self._dataset = self._dataset.skip(self.metadata['dataset_example_index']+1)

        print(f"Loaded state; Starting from step {self.metadata['dataset_example_index']}.", flush=True)

    def set_start_tokens(self, tokens):
        print(f'Dataset: setting start tokens to {tokens}')
        tokens_skipped = self.config.tokens_count_at_start # Might already have skipped tokens, so need to skip the difference
        self.config.tokens_count_at_start = tokens
        self._tokens_count_at_start = tokens
        self._total_tokens = tokens

        # if tokens - tokens_skipped > 0:
        #     print(f"Skipping {self.config.tokens_count_at_start} tokens at the start of the dataset.", flush=True)
        #     steps_to_skip = (tokens - tokens_skipped) // (self.config.seq_length * self.config.batch_size)
        #     self._dataset = self._dataset.skip(steps_to_skip)
        #     self.start_steps += steps_to_skip
        #     print(f"Skipped {self.start_steps} steps.", flush=True)



    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)



class OptHuggingfaceDataset(object):
    """ Optimized Huggingface dataset that loads pre-tokenized data from disk. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.pretokenized_dataset_dir = ''  # Directory of pre-tokenized dataset
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'
        config.tokens_count_at_start = 0
        config.throughput_average_window_size = 200

        # not used
        config.path = 'allenai/c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.tokenizer_processes = 4  # Number of parallel processes
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024

        return mlxu.update_config_dict(config, updates)

    @classmethod
    def load_dataset(cls, config):
        """
        Loads the pre-tokenized dataset from disk.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            datasets.Dataset or datasets.DatasetDict: Loaded dataset.
        """
        config = cls.get_default_config(config)
        if not os.path.isdir(config.pretokenized_dataset_dir):
            raise ValueError(f"Pre-tokenized dataset directory {config.pretokenized_dataset_dir} does not exist.")
        print(f'Loading {config.split} dataset from {config.pretokenized_dataset_dir}...')
        dataset = load_from_disk(config.pretokenized_dataset_dir)
        return dataset
    
    def set_start_tokens(self, tokens):
        print(f'Dataset: setting start tokens to {tokens}')
        self.config.tokens_count_at_start = tokens
        self._tokens_count_at_start = tokens
        self._total_tokens = tokens

    def __init__(self, config, tokenizer, text_processor):
        """
        Initializes the OptimizedHuggingfaceDataset.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = self.get_default_config(config)
        if not hasattr(self, 'config'):
            raise ValueError("Configuration is missing.")
        
        self._dataset = self.load_dataset(self.config)

        # dataset = Dataset.from_file("/n/home07/nabreu/SOO-LM/EasyLM/scratch/SOO-LM/tokenized/train/data-00000-of-05912.arrow")
        self._seq_length = self.config.seq_length
        self._batch_size = self.config.batch_size
        self._always_start_with_bos = self.config.always_start_with_bos
        self._batch_token_dtype = np.int32 if self.config.batch_token_dtype == 'i4' else np.int64
        self._tokens_count_at_start = self.config.tokens_count_at_start
        self._throughput_average_window_size = self.config.throughput_average_window_size

        # Initialize variables for batching
        self._chunk_size = self._batch_size * self._seq_length
        self._token_buffer = []
        self._loss_mask_buffer = []
        self._step_times = []
        self._start_time = time.time()
        self._total_tokens = self._tokens_count_at_start

        # Define BOS token ID if needed
        if self._always_start_with_bos:
            # Assuming you have access to the tokenizer's BOS token ID
            self._bos_token_id = tokenizer.bos_token_id
            if self._bos_token_id is None:
                raise ValueError("Tokenizer does not have a BOS token ID.")
        
        # If dataset is a DatasetDict, select the appropriate split
        if isinstance(self._dataset, dict):
            # Assuming 'train' split; modify if necessary
            self._dataset = self._dataset['train']

    def __iter__(self):
        """
        Iterator over the dataset that yields batches.

        Yields:
            tuple: (batch_dict, metrics_dict)
        """
        tokens_to_skip = self._tokens_count_at_start
        self._total_tokens = self._tokens_count_at_start  # Initialize total tokens

        # Iterator over the dataset
        dataset_iterator = iter(self._dataset)

        # Step 1: Skip tokens up to tokens_count_at_start
        while tokens_to_skip > 0:
            try:
                example = next(dataset_iterator)
            except StopIteration:
                # End of dataset reached before skipping desired tokens
                print("Reached end of dataset while skipping tokens.")
                return

            tokens = example['input_tokens']
            loss_masks = example['loss_masks']
            num_tokens = len(tokens)

            if tokens_to_skip >= num_tokens:
                # Skip the entire example
                tokens_to_skip -= num_tokens
                continue
            else:
                # Skip part of the example
                tokens = tokens[tokens_to_skip:]
                loss_masks = loss_masks[tokens_to_skip:]
                self._total_tokens += len(tokens)
                tokens_to_skip = 0  # All required tokens have been skipped

                # Add the remaining tokens to the buffer
                self._token_buffer.extend(tokens)
                self._loss_mask_buffer.extend(loss_masks)

        # Step 2: Continue iterating over the dataset and yielding batches
        for example in dataset_iterator:
            tokens = example['input_tokens']
            loss_masks = example['loss_masks']

            self._token_buffer.extend(tokens)
            self._loss_mask_buffer.extend(loss_masks)

            while len(self._token_buffer) > self._chunk_size + 1:
                # Prepare batch
                input_tokens = np.array(
                    self._token_buffer[:self._chunk_size],
                    dtype=self._batch_token_dtype
                ).reshape(self._batch_size, self._seq_length)

                target_tokens = np.array(
                    self._token_buffer[1:self._chunk_size + 1],
                    dtype=self._batch_token_dtype
                ).reshape(self._batch_size, self._seq_length)

                loss_masks = np.array(
                    self._loss_mask_buffer[1:self._chunk_size + 1],
                    dtype=np.float32
                ).reshape(self._batch_size, self._seq_length)

                if self._always_start_with_bos:
                    # Replace the first token with BOS token ID
                    input_tokens[:, 0] = self._bos_token_id

                # Yield the batch and metrics
                metrics = {
                    'dataset_total_tokens': self._total_tokens + self._chunk_size,
                }

                batch = {
                    'input_tokens': input_tokens,
                    'target_tokens': target_tokens,
                    'loss_masks': loss_masks,
                }

                yield batch, metrics

                # Update total tokens and remove the yielded tokens from the buffer
                self._total_tokens += self._chunk_size
                self._token_buffer = self._token_buffer[self._chunk_size:]
                self._loss_mask_buffer = self._loss_mask_buffer[self._chunk_size:]

    def get_state_dict(self):
        return {
            'config': self.config,
            'tokens_count_at_start': self._total_tokens,  # Save the current token count
        }

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))
        self._tokens_count_at_start = state_dict.get('tokens_count_at_start', self.config.tokens_count_at_start)
        self._total_tokens = self._tokens_count_at_start  # Ensure consistency

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def tokenizer(self):
        return None  # Not used in the optimized version

    @property
    def text_processor(self):
        return None  # Not used in the optimized version

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return self._dataset.features['input_tokens'].feature.dtype.size


# class OptHuggingfaceDataset(object):
#     """ Optimized Huggingface dataset that loads pre-tokenized data from disk. """

#     @staticmethod
#     def get_default_config(updates=None):
#         config = mlxu.config_dict()
#         config.pretokenized_dataset_dir = "/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/tokenized"  # Directory of pre-tokenized dataset
#         config.seq_length = 1024
#         config.batch_size = 8
#         config.always_start_with_bos = False
#         config.batch_token_dtype = 'i4'
#         config.tokens_count_at_start = 0
#         config.throughput_average_window_size = 200
        
        

#         # not used
#         config.path = 'allenai/c4'
#         config.name = 'en'
#         config.split = 'train'
#         config.streaming = False
#         config.tokenizer_processes = 4  # Number of parallel processes
#         config.tokenizer_parallel_chunk_size = 32
#         config.tokenizer_parallel_batch_size = 1024

#         return mlxu.update_config_dict(config, updates)

#     @classmethod
#     def load_dataset(cls, config):
#         """
#         Loads the pre-tokenized dataset from disk.

#         Args:
#             config (dict): Configuration dictionary.

#         Returns:
#             datasets.Dataset or datasets.DatasetDict: Loaded dataset.
#         """
#         config = cls.get_default_config(config)
#         if not os.path.isdir(config.pretokenized_dataset_dir):
#             raise ValueError(f"Pre-tokenized dataset directory {config.pretokenized_dataset_dir} does not exist.")
#         dataset = load_from_disk(config.pretokenized_dataset_dir)
#         # dataset = Dataset.from_file("/n/home07/nabreu/SOO-LM/EasyLM/scratch/SOO-LM/tokenized/train/data-00000-of-05912.arrow")
#         return dataset

#     def __init__(self, config, tokenizer, text_processor):
#         """
#         Initializes the OptimizedHuggingfaceDataset.

#         Args:
#             config (dict): Configuration dictionary.
#         """
#         self.config = self.get_default_config(config)
#         if not hasattr(self, 'config'):
#             raise ValueError("Configuration is missing.")
        
#         # TODO: move to flags
#         if self.config.split == 'validation':
#             self.config.pretokenized_dataset_dir = "/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/tokenized-val"

#         self._dataset = self.load_dataset(self.config)
#         self._seq_length = self.config.seq_length
#         self._batch_size = self.config.batch_size
#         self._always_start_with_bos = self.config.always_start_with_bos
#         self._batch_token_dtype = np.int32 if self.config.batch_token_dtype == 'i4' else np.int64
#         self._tokens_count_at_start = self.config.tokens_count_at_start
#         self._throughput_average_window_size = self.config.throughput_average_window_size

#         # Initialize variables for batching
#         self._chunk_size = self._batch_size * self._seq_length
#         self._token_buffer = []
#         self._loss_mask_buffer = []
#         self._step_times = []
#         self._start_time = time.time()
#         self._total_tokens = self._tokens_count_at_start

#         # If dataset is a DatasetDict, select the appropriate split
#         if isinstance(self._dataset, dict):
#             # Assuming 'train' split; modify if necessary
#             self._dataset = self._dataset['train']

#     def __iter__(self):
#         """
#         Iterator over the dataset that yields batches.

#         Yields:
#             tuple: (batch_dict, metrics_dict)
#         """
#         for example in self._dataset: # TODO: Start from self._tokens_count_at_start
#             tokens = example['input_tokens']
#             loss_masks = example['loss_masks']

#             self._token_buffer.extend(tokens)
#             self._loss_mask_buffer.extend(loss_masks)

#             while len(self._token_buffer) > self._chunk_size + 1:
#                 self._total_tokens += self._chunk_size
#                 metrics = {
#                     'dataset_total_tokens': self._total_tokens,
#                 }

#                 input_tokens = np.array(self._token_buffer[:self._chunk_size], dtype=self._batch_token_dtype).reshape(
#                     self._batch_size, self._seq_length
#                 )
#                 target_tokens = np.array(self._token_buffer[1:self._chunk_size + 1], dtype=self._batch_token_dtype).reshape(
#                     self._batch_size, self._seq_length
#                 )
#                 loss_masks = np.array(self._loss_mask_buffer[1:self._chunk_size + 1], dtype=np.float32).reshape(
#                     self._batch_size, self._seq_length
#                 )

#                 if self._always_start_with_bos:
#                     # Assuming BOS token ID is the first token in input_tokens
#                     input_tokens[:, 0] = self._dataset.features['input_tokens'].feature.int2str(
#                         self._dataset.features['input_tokens'].feature.dtype
#                     ).bos_token_id

#                 batch = {
#                     'input_tokens': input_tokens,
#                     'target_tokens': target_tokens,
#                     'loss_masks': loss_masks,
#                 }

#                 yield batch, metrics

#                 # Remove the chunk from the buffer
#                 self._token_buffer = self._token_buffer[self._chunk_size:]
#                 self._loss_mask_buffer = self._loss_mask_buffer[self._chunk_size:]

#     def get_state_dict(self):
#         return dict(config=self.config)

#     def load_state_dict(self, state_dict):
#         if 'config' in state_dict:
#             self.config.update(mlxu.ConfigDict(state_dict['config']))
#         self._tokens_count_at_start = state_dict.get('tokens_count_at_start', self.config.tokens_count_at_start)

#     @property
#     def seq_length(self):
#         return self._seq_length

#     @property
#     def tokenizer(self):
#         return None  # Not used in the optimized version

#     @property
#     def text_processor(self):
#         return None  # Not used in the optimized version

#     @property
#     def dataset(self):
#         return self._dataset

#     @property
#     def vocab_size(self):
#         return self._dataset.features['input_tokens'].feature.dtype.size



class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(mlxu.ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
