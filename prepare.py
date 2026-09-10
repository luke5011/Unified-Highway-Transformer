import os
import numpy as np
import tiktoken
from datasets import load_dataset

def main():
    print("Downloading WikiText-103...")
    # Load the raw WikiText-103 dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    # Initialize the GPT-2 tokenizer to match train.py
    enc = tiktoken.get_encoding("gpt2")
    
    # Process both train and validation splits
    splits = [('train', dataset['train']), ('val', dataset['validation'])]
    
    for split_name, split_data in splits:
        print(f"\nProcessing {split_name} split...")
        
        all_ids = []
        # Iterate through the text, encode it, and append to our master list
        for text in split_data['text']:
            if text.strip():  # Skip completely empty structural lines
                ids = enc.encode(text)
                all_ids.extend(ids)
        
        print(f"Total tokens in {split_name}: {len(all_ids):,}")
        
        # Save to binary format (uint16 safely holds the 50,257 GPT-2 vocab)
        filename = f"wiki_{split_name}.bin"
        arr = np.array(all_ids, dtype=np.uint16)
        
        # Write directly to disk
        arr.tofile(filename)
        print(f"Saved to {filename}")

if __name__ == '__main__':
    main()