import json
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = []  # Stores the final vocabulary as a list of tokens
        self.merges = {} # Stores the learned merge operations: (token1, token2) -> merged_token

    def train(self, proto_token_counts: dict[str, int]):
        """
        Trains the BPE model on a corpus of proto-tokens and their frequencies.
        proto_token_counts: A dictionary mapping proto-tokens (strings) to their counts.
        """
        if not proto_token_counts:
            print("Warning: Training corpus is empty.")
            return

        # 1. Initialize vocabulary with all unique characters from the proto-tokens.
        initial_char_vocab = set()
        for token_str in proto_token_counts.keys():
            for char in token_str:
                initial_char_vocab.add(char)
        
        self.vocab = sorted(list(initial_char_vocab))
        
        # Represent each proto-token as a list of its current sub-tokens (initially characters)
        # e.g., "text-blue" with count 5 becomes (['t', 'e', 'x', 't', '-', 'b', 'l', 'u', 'e'], 5)
        # This list will be modified as merges happen.
        current_token_sequences = {
            word: list(word) for word in proto_token_counts.keys()
        }

        num_merges_needed = self.vocab_size - len(self.vocab)

        for i in range(num_merges_needed):
            # a. Calculate frequencies of adjacent pairs in the current token sequences
            pair_freqs = defaultdict(int)
            for word, count in proto_token_counts.items():
                sequence = current_token_sequences[word]
                for k in range(len(sequence) - 1):
                    pair_freqs[(sequence[k], sequence[k+1])] += count
            
            if not pair_freqs:
                # print(f"No more pairs to merge after {i} merges.")
                break # No more pairs to merge

            # b. Find the most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # c. Create the new merged token
            new_token = best_pair[0] + best_pair[1]
            
            # Add to vocabulary and store merge rule
            self.vocab.append(new_token)
            self.merges[best_pair] = new_token
            
            # d. Update all current_token_sequences by applying the new merge
            for word in current_token_sequences: # Iterate over keys (original proto-tokens)
                old_sequence = current_token_sequences[word]
                
                j = 0
                updated_sequence = []
                while j < len(old_sequence):
                    if j < len(old_sequence) - 1 and \
                       (old_sequence[j], old_sequence[j+1]) == best_pair:
                        updated_sequence.append(new_token)
                        j += 2
                    else:
                        updated_sequence.append(old_sequence[j])
                        j += 1
                current_token_sequences[word] = updated_sequence
        # print(f"Training complete. Vocab size: {len(self.vocab)}")

    def tokenize(self, pre_segmented_tokens: list[str]) -> list[str]:
        """
        Tokenizes a list of pre-segmented proto-tokens using the learned BPE model.
        Returns a flat list of BPE tokens.
        """
        final_bpe_tokens = []
        for proto_token in pre_segmented_tokens:
            if not proto_token: # Skip empty strings if any
                continue

            # Start with the proto-token split into characters
            current_sub_tokens = list(proto_token)
            
            # Iteratively apply learned merges
            while True:
                can_merge = False
                k = 0
                merged_sequence = []
                while k < len(current_sub_tokens):
                    if k < len(current_sub_tokens) - 1:
                        pair = (current_sub_tokens[k], current_sub_tokens[k+1])
                        if pair in self.merges:
                            merged_sequence.append(self.merges[pair])
                            k += 2
                            can_merge = True
                        else:
                            merged_sequence.append(current_sub_tokens[k])
                            k += 1
                    else: # Last token
                        merged_sequence.append(current_sub_tokens[k])
                        k += 1
                
                current_sub_tokens = merged_sequence
                if not can_merge: # No more merges could be applied in this pass
                    break
            final_bpe_tokens.extend(current_sub_tokens)
            
        return final_bpe_tokens

    def save_model(self, path: str):
        """Saves the learned vocabulary and merges to a JSON file."""
        # Convert tuple keys in merges to strings for JSON serialization
        serializable_merges = {f"{p[0]}'''{p[1]}": m for p, m in self.merges.items()}
        model_data = {
            "vocab": self.vocab,
            "merges": serializable_merges,
            "vocab_size": self.vocab_size
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2)
            # print(f"BPE model saved to {path}")
        except IOError as e:
            print(f"Error saving BPE model to {path}: {e}")


    def load_model(self, path: str):
        """Loads a pre-trained BPE model from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.vocab = model_data.get("vocab", [])
            serializable_merges = model_data.get("merges", {})
            # Convert string keys back to tuples
            self.merges = {}
            for k_str, v_merged_token in serializable_merges.items():
                parts = k_str.split("'''", 1) # Split only on the first occurrence of '''
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = v_merged_token
                else:
                    print(f"Warning: Could not parse merge key '{k_str}' from loaded model.")

            self.vocab_size = model_data.get("vocab_size", len(self.vocab) + len(self.merges)) # Estimate if not saved
            # print(f"BPE model loaded from {path}. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")
        except FileNotFoundError:
            print(f"Error: BPE model file not found at {path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from BPE model file {path}")
        except IOError as e:
            print(f"Error loading BPE model from {path}: {e}")

if __name__ == '__main__':
    # Example usage (conceptual, expanded for testing)
    # This part is for direct testing of bpe_tokenizer.py
    
    print("--- BPE Tokenizer Direct Test ---")
    
    # 1. Create some sample proto-token counts (as if from HTMLTailwindPreprocessor)
    sample_proto_counts = Counter({
        "<div>": 10,
        "class=": 8,
        "text-blue-500": 5,
        "font-bold": 5,
        "Hello": 3,
        "World": 3,
        "</div>": 10,
        "<p>": 7,
        "example": 4,
        "</p>": 7
    })
    
    # 2. Train the BPE tokenizer
    # Let's aim for a small vocab size for this test to see some merges
    # Initial chars: d,i,v,<,>,c,l,a,s,=,t,e,x,-,b,u,5,0,f,o,n,H,W,r,p,m
    # (~25 initial chars)
    # Let's target vocab_size = 35, so about 10 merges.
    bpe_trainer = BPETokenizer(vocab_size=40) 
    print(f"Training BPE with vocab_size={bpe_trainer.vocab_size}...")
    bpe_trainer.train(sample_proto_counts)
    print(f"Training complete. Final vocab size: {len(bpe_trainer.vocab)}")
    print(f"Learned merges: {bpe_trainer.merges}")

    # 3. Test tokenization
    sample_pre_segmented = ["<div", "class=", "text-blue-500", "font-bold", ">", "Hello", "World", "</div>"]
    print(f"\nOriginal pre-segmented: {sample_pre_segmented}")
    bpe_tokens = bpe_trainer.tokenize(sample_pre_segmented)
    print(f"BPE Tokens: {bpe_tokens}")

    # 4. Test save and load
    model_path = "test_bpe_model.json"
    print(f"\nSaving BPE model to {model_path}...")
    bpe_trainer.save_model(model_path)

    bpe_loader = BPETokenizer()
    print(f"Loading BPE model from {model_path}...")
    bpe_loader.load_model(model_path)
    
    print(f"Loaded model vocab size: {len(bpe_loader.vocab)}")
    print(f"Loaded model merges: {bpe_loader.merges}")

    loaded_bpe_tokens = bpe_loader.tokenize(sample_pre_segmented)
    print(f"Tokens from loaded model: {loaded_bpe_tokens}")

    # Assert that loaded model gives same tokenization
    # This requires merges to be deterministically learned if multiple have same freq,
    # or for the test case to be simple enough not to hit that.
    # For this example, it should be fairly stable.
    if bpe_tokens == loaded_bpe_tokens:
        print("\nTokenization consistent between trained and loaded model. Test PASSED.")
    else:
        print("\nTokenization INCONSISTENT. Test FAILED.")
        print(f"Original model tokens: {bpe_tokens}")
        print(f"Loaded model tokens:   {loaded_bpe_tokens}")

    # Clean up the test file
    import os
    try:
        os.remove(model_path)
        print(f"Cleaned up {model_path}")
    except OSError:
        pass
    print("--- BPE Tokenizer Direct Test End ---") 