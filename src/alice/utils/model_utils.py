"Code for model saving and checkpointing."
import os
import torch
from alice.bo.optimizer import BotorchOptimizer

def save_checkpoint(bo_optimizer: BotorchOptimizer, iteration: int, base_dir: str="./checkpoints"):
    """
    Save a model checkpoint: the LoRA adapters plus the trained projection heads.
    Args:
        bo_optimizer (BotorchOptimizer):  Optimizer object for BO
        iteration (int): Current iteration number for checkpoint naming
        base_dir (str): Base directory for saving checkpoints
    """
    checkpoint_dir = os.path.join(base_dir, f"iteration_{iteration}")
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\nSaving model components to: {checkpoint_dir}")
        featurizer = bo_optimizer.surrogate_model.finetuning_model

        adapter_save_path = os.path.join(checkpoint_dir, "adapters")
        featurizer.llm.save_pretrained(adapter_save_path)
        print(f"Adapters saved to: {adapter_save_path}")

        if hasattr(featurizer, "projection_heads"):
            heads_save_path = os.path.join(checkpoint_dir, "projection_heads.pth")
            torch.save(featurizer.projection_heads.state_dict(), heads_save_path)
            print(f"Projection heads saved to: {heads_save_path}")

        if hasattr(featurizer, "attention_layer"):
            attn_save_path = os.path.join(checkpoint_dir, "attention_layer.pth")
            torch.save(featurizer.attention_layer.state_dict(), attn_save_path)
            print(f"Attention layer saved to: {attn_save_path}")

        print("Models saved!")

    except AttributeError as e:
        print(f"Could not save model. A model component was not found: {e}")

    except Exception as e:
        print(f"Error during saving: {e}")
