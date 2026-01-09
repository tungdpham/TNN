import os
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

# 1. Create the target directory
target_dir = "data/open-web-text"
os.makedirs(target_dir, exist_ok=True)

# 2. Load the dataset (using streaming if you have low RAM, or normal if you have 64GB+)
# OpenWebText doesn't have a formal 'validation' split on HF, so we usually split it manually
dataset = load_dataset("skylion007/openwebtext", num_proc=8)

# 3. Setup Tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# 4. Tokenize
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=8,
)

# 5. Export to binary (Fixed Version)
for split, dset in tokenized.items():
    filename = os.path.join(target_dir, f"{split}.bin")
    
    total_batches = 1024
    print(f"Writing {filename}...")
    
    with open(filename, 'wb') as f:
        for batch_idx in tqdm(range(total_batches)):
            # Get a shard of the data
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            if len(batch) == 0: continue
            
            # THE FIX: Cast each list to a numpy array of uint16 before concatenating
            # This bypasses the casting rule error
            batch_arrays = [np.array(ids, dtype=np.uint16) for ids in batch['ids']]
            flat_ids = np.concatenate(batch_arrays)
            
            f.write(flat_ids.tobytes())

    print(f"Finished saving {split} to {target_dir}")