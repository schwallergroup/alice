"""Module for LoRA fine-tuned LLM embeddings"""
from abc import abstractmethod
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm

from peft import LoraConfig, get_peft_model

from alice.featurization.utils.pooling import average_pool, last_token_pool, weighted_average_pool
from alice.featurization.text import get_model_and_tokenizer
from alice.featurization.utils.layers import get_target_layers


class TaskAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 1. Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 2. Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # 3. Dropout regularization
        self.dropout = nn.Dropout(dropout)

        # 4. Learnable gating parameter
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        gating_factor = torch.tanh(self.gate)
        x = self.norm(x + (gating_factor * self.dropout(attn_output)))
        return x


class BaseNNFeaturizer(nn.Module):
    """
    Base class for neural network-based featurizers.
    """
    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim

    @property
    def output_dim(self) -> int:
        """
        Returns the output dimension of the featurizer.
        
        Returns:
            int: Output dimension
        """
        return self._output_dim

    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        pass


class ConfigurableMLP(BaseNNFeaturizer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128],
        activation: str = "elu",
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        """
        Configurable MLP for projecting embeddings to lower dimensions.

        Args:
            input_dim: Input feature dimension (embedding dimensions)
            output_dim: Output projection dimension
            hidden_dims: List of hidden layer sizes (empty list = single linear layer)
            activation: Activation function ('elu', 'relu', or 'gelu')
            use_layer_norm: Whether to apply LayerNorm after each linear layer
            dropout: Dropout probability (applied after activations)
            use_spectral_norm: Whether to constrain weights with spectral normalization
        """
        super().__init__(input_dim=input_dim, projection_dim=output_dim)

        layers = []
        curr_dim = input_dim

        for h_dim in hidden_dims:
            linear_layer = nn.Linear(curr_dim, h_dim)
            if use_spectral_norm:
                linear_layer = spectral_norm(linear_layer)

            layers.append(linear_layer)

            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))

            if activation == "elu":
                layers.append(nn.ELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            curr_dim = h_dim

        final_layer = nn.Linear(curr_dim, output_dim)
        if use_spectral_norm:
            final_layer = spectral_norm(final_layer)

        layers.append(final_layer)

        self.net = nn.Sequential(*layers)
        self._output_dim = output_dim

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                if hasattr(m, 'weight'):
                    init.xavier_uniform_(m.weight)

    def forward(self, x):
        dummy_param = next(self.net.parameters())
        x = x.to(dummy_param.dtype)
        return self.net(x)


class LLMFeaturizer(BaseNNFeaturizer):
    def __init__(
        self,
        model_name: str = "t5-base",
        input_dim: int = 768,
        projection_dim: Optional[int] = None,
        trainable: bool = True,
        pooling_method: str = "average",
        normalize_embeddings: bool = False,
        lora_dropout: float = 0.2,
        modules_to_save: Optional[List[str]] = ["head"],
        target_ratio: float = 0.25,
        from_top: bool = True,
        num_objectives: int = 2,
        use_task_attention: bool = False,
        independent_projection_heads: bool = True,
        projection_kwargs: Optional[Dict[str, Any]] = None,
    ):
        proj_config = projection_kwargs or {
            "hidden_dims": [128],
            "activation": "elu", 
            "dropout": 0.1,
            "use_spectral_norm": True
        }

        if use_task_attention and not independent_projection_heads:
            print("[WARN] Task Attention requires independent projection heads. Forcing independent_projection_heads=True.")
            independent_projection_heads = True

        if independent_projection_heads is True:
            final_output_dim = projection_dim * num_objectives if projection_dim is not None else input_dim
        else:
            final_output_dim = projection_dim if projection_dim is not None else input_dim

        base_proj_dim = projection_dim if projection_dim is not None else input_dim
        super().__init__(input_dim=input_dim, projection_dim=base_proj_dim)

        self.num_objectives = num_objectives
        self.embedding_dim = input_dim
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.input_dim = input_dim
        self.trainable = trainable
        self.use_task_attention = use_task_attention
        self.independent_projection_heads = independent_projection_heads

        print("\n" + "="*60)
        print("[Featurizer] Initializing Deep LLM Featurizer")
        print(f"   - Base Model:       {model_name}")
        print(f"   - Trainable:        {trainable}")
        print(f"   - Pooling:          {pooling_method}")
        print(f"   - Input Dim:        {input_dim}")

        if projection_dim:
            print(f"   - Projection Dim:   {projection_dim} (per objective)")
            print(f"   - Num Objectives:   {num_objectives}")
            print(f"   - Strategy:         {'INDEPENDENT Heads (N separate networks)' if independent_projection_heads else 'SHARED Head (1 network)'}")
            print(f"   - Task Attention:   {'ENABLED (Objectives interact)' if use_task_attention else 'DISABLED'}")
            print(f"   - MLP Architecture: {proj_config}")
        else:
            print("   - Projection:       None (Using raw embeddings)")
        print("="*60 + "\n")

        self.llm, self.tokenizer = get_model_and_tokenizer(model_name, "cuda")

        if trainable:
            target_modules = get_target_layers(self.llm, target_ratio, from_top)
            self.llm = get_peft_model(
                self.llm,
                LoraConfig(
                    r=4,
                    lora_alpha=16,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    use_rslora=True,
                    modules_to_save=modules_to_save,
                ),
            )
            self.llm.print_trainable_parameters()
        else:
            self.llm.requires_grad_(False)

        if projection_dim is not None:
            if self.independent_projection_heads:
                print(f"[Featurizer] Creating {num_objectives} INDEPENDENT heads with config: {proj_config}")
                self.projection_heads = nn.ModuleList([
                    ConfigurableMLP(
                        input_dim=input_dim,
                        output_dim=projection_dim,
                        **proj_config
                    )
                    for _ in range(num_objectives)
                ])
            else:
                print(f"[Featurizer] Creating 1 SHARED head with config: {proj_config}")
                self.projection_heads = nn.ModuleList([
                    ConfigurableMLP(
                        input_dim=input_dim,
                        output_dim=projection_dim,
                        **proj_config
                    )
                ])

            print(f"\n[Featurizer] Projection Head Architecture:\n{self.projection_heads}")
            if self.use_task_attention:
                print("[MODEL] Enabling Task Attention Layer (Objectives will interact)")
                self.attention_layer = TaskAttentionLayer(embed_dim=projection_dim)

            self._output_dim = final_output_dim
        else:
            self._output_dim = input_dim # raw_embeddings

    def get_embeddings(self, x, batch_size=4):
        torch.cuda.empty_cache()

        x = x.to(dtype=torch.float32)
        self.llm = self.llm.to(dtype=torch.float32)

        n_points = x.size(0)
        ids_split = int(x.shape[-1] / 2)

        embedding_chunks = []

        for start_idx in range(0, n_points, batch_size):

            torch.cuda.empty_cache()
            end_idx = min(start_idx + batch_size, n_points)
            input_ids = x[start_idx:end_idx, :ids_split].long()
            attn_mask = x[start_idx:end_idx, ids_split:].long()

            if self.trainable:
                outputs = self.llm(
                    input_ids=input_ids, attention_mask=attn_mask
                )

            else:
                self.llm.eval()
                with torch.no_grad():
                    outputs = self.llm(
                        input_ids=input_ids, attention_mask=attn_mask
                    )

            last_hidden_state = outputs.last_hidden_state

            if self.pooling_method == "average":
                pooled = average_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "cls":
                pooled = last_hidden_state[:, 0]
            elif self.pooling_method == "last_token_pool":
                pooled = last_token_pool(last_hidden_state, attn_mask)
            elif self.pooling_method == "weighted_average":
                pooled = weighted_average_pool(last_hidden_state, attn_mask)
            else:
                raise ValueError(
                    f"Unknown pooling method: {self.pooling_method}"
                )

            if self.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)

            embedding_chunks.append(pooled.to(dtype=torch.float32))
            del outputs, last_hidden_state, pooled
            torch.cuda.empty_cache()

        embeddings = torch.cat(embedding_chunks, dim=0)
        return embeddings

    def forward(self, x):
        if x.dim() == 2:
            shared_embeddings = self.get_embeddings(x)
        elif x.dim() == 3:
            n_candidates, n_train, _ = x.shape
            train_data = x[0, : n_train - 1, :]
            all_candidates = x[:, n_train - 1, :]
            with torch.no_grad():
                train_embeddings = self.get_embeddings(train_data)
                all_candidate_embeddings = self.get_embeddings(all_candidates)

            train_embeddings = train_embeddings.unsqueeze(0).expand(
                n_candidates, -1, -1
            )
            candidate_embeddings = all_candidate_embeddings.unsqueeze(1)
            shared_embeddings = torch.cat(
                [train_embeddings, candidate_embeddings], dim=1
            )
        else:
            raise ValueError("Input dimension must be 2D or 3D.")

        if hasattr(self, 'projection_heads'):
            projection_outputs = [
                head(shared_embeddings) for head in self.projection_heads
            ]

            if self.use_task_attention:
                # apply cross objective self attention by stacking different projectsion on top of each other
                stacked_features = torch.stack(projection_outputs, dim=1)
                aligned_features = self.attention_layer(stacked_features)
                return torch.flatten(aligned_features, start_dim=1)

            return torch.cat(projection_outputs, dim=-1)

        return shared_embeddings # no projection_heads fallback

