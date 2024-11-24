import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_unflatten
import numpy as np
from transformers import AutoTokenizer
from typing import Dict, Optional, Tuple
import math
from dataclasses import dataclass
from pathlib import Path
import os
import time
from datetime import datetime, timedelta
import random
import pickle


@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def _norm(self, x):
        return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(mx.float32)).astype(x.dtype)
        return self.weight * output


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads

        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = None
        self.cache_v = None

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.reshape(B, L, self.n_local_heads, self.head_dim)
        k = k.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)  # (B, H, L, D)
        k = k.transpose(0, 2, 1, 3)  # (B, H, L, D)
        v = v.transpose(0, 2, 1, 3)  # (B, H, L, D)

        # Flash attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        if mask is not None:
            # Reshape mask for broadcasting: (B, 1, 1, L) to match scores shape (B, H, L, L)
            mask = mask.reshape(B, 1, 1, L)
            mask = mx.where(mask == 0, float('-inf'), 0.0)
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1)
        output = scores @ v  # (B, H, L, D)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = x + self.attention(self.attention_norm(x), mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        output = self.output(h)
        return output


def build_model(model_path: str) -> Tuple[ModelArgs, Transformer]:
    model_path = Path(model_path)
    weights = mx.load(str(model_path / "weights.npz"))
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
        config.pop("model_type", None)
        args = ModelArgs(**config)

    model = Transformer(args)
    model.update(tree_unflatten(list(weights.items())))
    return args, model


def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def update_training_state(epoch, samples_processed, total_samples, epoch_loss, batch_count, model, start_time, num_epochs):
    try:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Calculate metrics
        progress_percent = (samples_processed / total_samples) * 100
        current_loss = epoch_loss / batch_count if batch_count > 0 else 0
        samples_per_second = samples_processed / elapsed_time if elapsed_time > 0 else 0
        tokens_per_second = samples_per_second * model.args.max_seq_len
        
        # Get actual memory usage from process
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
        except:
            memory_mb = 0
            
        # Save metrics to a log file
        metrics = {
            'timestamp': current_time,
            'epoch': epoch,
            'progress_percent': progress_percent,
            'current_loss': float(current_loss),
            'samples_per_second': samples_per_second,
            'tokens_per_second': tokens_per_second,
            'memory_mb': memory_mb,
            'model_config': {
                'dim': model.args.dim,
                'n_layers': model.args.n_layers,
                'n_heads': model.args.n_heads,
                'max_seq_len': model.args.max_seq_len,
            }
        }
        
        with open('training_metrics.jsonl', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
        return {
            'progress_percent': progress_percent,
            'current_loss': current_loss,
            'samples_per_second': samples_per_second,
            'tokens_per_second': tokens_per_second,
            'memory_mb': memory_mb
        }
    except Exception as e:
        print(f"Error updating training state: {e}")
        return None


def save_model(model, filepath):
    try:
        # Convert parameters to basic MLX arrays
        params = {}
        for k, v in model.parameters().items():
            if hasattr(v, 'array'):
                params[k] = v.array
            elif hasattr(v, 'value'):
                params[k] = v.value
            else:
                params[k] = v
        
        # Try MLX native save
        mx.savez(filepath, **params)
        print(f"Model saved to {filepath}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        # Backup save with pickle
        backup_path = filepath + '.backup'
        try:
            with open(backup_path, 'wb') as f:
                pickle.dump(params, f)  # Save the converted params
            print(f"Created backup save at {backup_path}")
        except Exception as e2:
            print(f"Backup save also failed: {e2}")


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02.0f}:{m:02.0f}:{s:02.0f}"


def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Load and process dataset
    train_data = load_dataset("training_data.json")
    total_samples = len(train_data)
    
    # Optimized model configuration for M4 Mac Mini 16GB
    model_args = ModelArgs(
        dim=768,             # Increased from 512 for better capacity
        n_layers=12,         # Increased from 8 for deeper learning
        n_heads=12,          # Increased from 8 for better attention
        vocab_size=len(tokenizer),
        max_seq_len=1024     # Increased from 512 for longer context
    )
    
    # Initialize model and optimizer
    model = Transformer(model_args)
    optimizer = optim.Adam(learning_rate=1e-4)  # Increased for faster learning
    
    # Training parameters
    batch_size = 4          # Increased from 2 for better parallelization
    epochs_per_cycle = 25   # Increased from 10 for longer training
    patience = 3           # Number of epochs to wait for improvement
    min_improvement = 0.01  # Minimum loss improvement to count as progress
    checkpoint_interval = 30 * 60  # Save every 30 minutes
    
    start_time = time.time()
    last_save_time = start_time
    samples_processed = 0
    epoch = 0
    
    # Early stopping variables
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = 'best_model.npz'

    try:
        # Main training loop
        while epoch < epochs_per_cycle and epochs_without_improvement < patience:
            epoch += 1
            random.shuffle(train_data)
            print(f"\nEpoch {epoch}/{epochs_per_cycle}")
            epoch_loss = 0.0
            batch_count = 0
            
            # Process data in batches
            for i in range(0, len(train_data), batch_size):
                try:
                    batch = train_data[i:i + batch_size]
                    batch_inputs = []
                    batch_targets = []
                    
                    # Prepare batch data
                    for item in batch:
                        # Format the prompt
                        prompt = f"### Instruction: {item['instruction']}\n\n### Input: {item['input']}\n\n### Response: {item['output']}"
                        
                        # Tokenize input and target
                        tokens = tokenizer.encode(prompt)
                        if len(tokens) > model_args.max_seq_len:
                            tokens = tokens[:model_args.max_seq_len]
                        
                        # Create input/target pairs for causal language modeling
                        input_ids = tokens[:-1]  # All tokens except last
                        target_ids = tokens[1:]  # All tokens except first
                        
                        batch_inputs.append(input_ids)
                        batch_targets.append(target_ids)
                    
                    # Pad sequences to same length
                    max_len = max(len(ids) for ids in batch_inputs)
                    batch_inputs = [ids + [0] * (max_len - len(ids)) for ids in batch_inputs]
                    batch_targets = [ids + [0] * (max_len - len(ids)) for ids in batch_targets]
                    
                    # Convert to MLX arrays
                    input_array = mx.array(batch_inputs)
                    target_array = mx.array(batch_targets)
                    
                    # Forward pass and loss calculation
                    def loss_fn(model, x, y):
                        logits = model(x)  # [batch, seq_len, vocab_size]
                        # Reshape logits and targets for cross entropy
                        logits = logits.reshape(-1, model.vocab_size)  # [batch*seq_len, vocab_size]
                        targets = y.reshape(-1)  # [batch*seq_len]
                        # Create a mask to ignore padding tokens
                        mask = (targets != 0).astype(mx.float32)
                        # Calculate masked cross entropy loss
                        ce_loss = nn.losses.cross_entropy(logits, targets, reduction='none')
                        masked_loss = (ce_loss * mask).sum() / mask.sum()
                        return masked_loss
                    
                    # Compute loss and gradients
                    loss, grads = nn.value_and_grad(model, loss_fn)(model, input_array, target_array)
                    optimizer.update(model, grads)
                    mx.eval(model.parameters())  # Force sync
                    
                    # Update training statistics
                    epoch_loss += float(loss)
                    batch_count += 1
                    samples_processed += len(batch)
                    
                    # Check if it's time to save (every 30 minutes)
                    current_time = time.time()
                    if current_time - last_save_time >= checkpoint_interval:
                        save_model(model, "model_checkpoint.pkl")
                        last_save_time = current_time
                    
                    # Update training state less frequently (every 50 batches)
                    if batch_count % 50 == 0:
                        state = update_training_state(
                            epoch=epoch,
                            samples_processed=samples_processed,
                            total_samples=total_samples,
                            epoch_loss=epoch_loss,
                            batch_count=batch_count,
                            model=model,
                            start_time=start_time,
                            num_epochs=epoch + 1
                        )
                        
                        if state:
                            print(f"Epoch {epoch}/{epochs_per_cycle} | "
                                  f"Progress: {state['progress_percent']:.1f}% | "
                                  f"Loss: {state['current_loss']:.4f} | "
                                  f"Speed: {state['samples_per_second']:.1f} samples/s | "
                                  f"Tokens/s: {state['tokens_per_second']:.1f}")
                    
                    # Memory cleanup
                    if batch_count % 50 == 0:
                        mx.eval(None)  # Clear unused memory
                    
                except Exception as batch_error:
                    print(f"Error processing batch: {batch_error}")
                    continue  # Skip problematic batch
            
            # End of epoch processing
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
            
            # Check for improvement
            if avg_epoch_loss < best_loss - min_improvement:
                print(f"Loss improved from {best_loss:.4f} to {avg_epoch_loss:.4f}")
                best_loss = avg_epoch_loss
                epochs_without_improvement = 0
                # Save best model
                save_model(model, best_model_path)
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs. Best loss: {best_loss:.4f}")
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered!")
                    break
            
            # Save at end of epoch
            save_model(model, 'mlx_model_best.npz')
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_model(model, 'mlx_model_best.npz')
    except Exception as e:
        print(f"\nError during training: {e}")
        save_model(model, 'mlx_model_best.npz')
    finally:
        # Final state update
        update_training_state(
            epoch=epoch,
            samples_processed=samples_processed,
            total_samples=total_samples,
            epoch_loss=epoch_loss,
            batch_count=batch_count,
            model=model,
            start_time=start_time,
            num_epochs=epoch + 1
        )


if __name__ == "__main__":
    main()
