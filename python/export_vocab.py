import tiktoken
import struct
import os

def export_vocab():
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    output_path = "data/open-web-text/vocab.bin"
    
    print(f"Exporting vocab size: {vocab_size} to {output_path}")
    
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", vocab_size))
        for i in range(vocab_size):
            token_bytes = enc.decode_bytes([i])
            # write length (uint32) then bytes
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)
            
    print("Done.")

if __name__ == "__main__":
    export_vocab()
