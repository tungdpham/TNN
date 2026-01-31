import os
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

target_dir = "data/open-web-text"
os.makedirs(target_dir, exist_ok=True)

dataset = load_dataset("skylion007/openwebtext", num_proc=8)

enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=8,
)

for split, dset in tokenized.items():
    filename = os.path.join(target_dir, f"{split}.bin")
    
    total_batches = 1024
    print(f"Writing {filename}...")
    
    with open(filename, 'wb') as f:
        for batch_idx in tqdm(range(total_batches)):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            if len(batch) == 0: continue

            batch_arrays = [np.array(ids, dtype=np.uint16) for ids in batch['ids']]
            flat_ids = np.concatenate(batch_arrays)
            
            f.write(flat_ids.tobytes())

    print(f"Finished saving {split} to {target_dir}")