import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np
import pickle


class MultiHeadMLP(nn.Module):
    """
    Multi-Head MLP for QRT Challenge
    
    Architecture:
    - Shared encoder: processes illiquid asset returns (RET_0 to RET_99)
    - Multiple heads: one for each liquid asset (ID_TARGET 0-99)
    - Each head outputs a single logit for binary classification
    """
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [256, 128, 64], 
                 num_heads: int = 100, dropout: float = 0.2, activation: str = 'relu',
                 industry_num_classes: Optional[Dict[str, int]] = None, industry_embedding_dim: Optional[Dict[str, int]] = None):
        """
        Args:
            input_dim: number of input features (default: 100 for RET_0 to RET_99)
            hidden_dims: list of hidden layer dimensions for shared encoder
            num_heads: number of output heads (default: 100 for liquid assets 0-99)
            dropout: dropout rate
            activation: activation function ('relu', 'leaky_relu', 'tanh')
        """
        super(MultiHeadMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        
        # 行业embedding
        self.industry_cols = list(industry_num_classes.keys()) if industry_num_classes else []
        self.industry_num_classes = industry_num_classes or {}
        # 自动分配embedding_dim
        self.industry_embedding_dim = industry_embedding_dim or {col: min(16, n//2+1) for col, n in self.industry_num_classes.items()}
        self.industry_embeds = nn.ModuleDict({
            col: nn.Embedding(self.industry_num_classes[col], self.industry_embedding_dim[col])
            for col in self.industry_cols
        })
        # 计算总输入维度
        total_input_dim = input_dim + sum(self.industry_embedding_dim.values())
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build shared encoder
        encoder_layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*encoder_layers)
        
        # Create individual heads for each liquid asset
        self.heads = nn.ModuleDict()
        for i in range(num_heads):
            self.heads[f'head_{i}'] = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, industry_ids: torch.Tensor, id_targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: input features [batch_size, input_dim]
            id_targets: target asset IDs [batch_size]
        
        Returns:
            logits: [batch_size] - one logit per sample
        """
        # x: [batch, input_dim], industry_ids: [batch, n_industry], id_targets: [batch]
        embed_list = []
        if self.industry_cols:
            for i, col in enumerate(self.industry_cols):
                embed = self.industry_embeds[col](industry_ids[:, i])  # [batch, emb_dim]
                embed_list.append(embed)
        if embed_list:
            x = torch.cat([x] + embed_list, dim=1)
        shared_features = self.shared_encoder(x)  # [batch_size, last_hidden_dim]
        
        # 向量化选择head
        # 1. 先把所有head的输出拼成一个 [batch_size, num_heads] 矩阵
        all_head_outputs = []
        for i in range(self.num_heads):
            all_head_outputs.append(self.heads[f'head_{i}'](shared_features))  # [batch_size, 1]
        all_head_outputs = torch.cat(all_head_outputs, dim=1)  # [batch_size, num_heads]
        # 2. 用id_targets做gather，选出每个样本对应的head输出
        logits = all_head_outputs.gather(1, id_targets.view(-1, 1)).squeeze(1)  # [batch_size]
        return logits
    
    def predict_proba(self, x: torch.Tensor, id_targets: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions
        
        Args:
            x: input features
            id_targets: target asset IDs
        
        Returns:
            probabilities: [batch_size] - probability of positive class
        """
        logits = self.forward(x, id_targets)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, id_targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions
        
        Args:
            x: input features
            id_targets: target asset IDs
            threshold: classification threshold
        
        Returns:
            predictions: [batch_size] - binary predictions (0 or 1)
        """
        probs = self.predict_proba(x, id_targets)
        return (probs > threshold).float()
    
    def get_head_parameters(self, head_id: int) -> Dict[str, torch.Tensor]:
        """Get parameters of a specific head"""
        head_name = f'head_{head_id}'
        if head_name in self.heads:
            return {name: param.clone() for name, param in self.heads[head_name].named_parameters()}
        return {}
    
    def get_shared_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters of the shared encoder"""
        return {name: param.clone() for name, param in self.shared_encoder.named_parameters()}


class MultiTaskMLP(nn.Module):
    """
    Alternative Multi-Task MLP with shared encoder and task-specific decoders
    
    This version uses a more sophisticated multi-task learning approach
    """
    
    def __init__(self, input_dim: int = 100, encoder_dims: List[int] = [256, 128], 
                 decoder_dims: List[int] = [64, 32], num_heads: int = 100, 
                 dropout: float = 0.2):
        super(MultiTaskMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.shared_encoder = nn.Sequential(*encoder_layers)
        
        # Task-specific decoders
        self.decoders = nn.ModuleDict()
        for i in range(num_heads):
            decoder_layers = []
            prev_decoder_dim = prev_dim
            for dim in decoder_dims:
                decoder_layers.extend([
                    nn.Linear(prev_decoder_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_decoder_dim = dim
            
            # Final output layer
            decoder_layers.append(nn.Linear(prev_decoder_dim, 1))
            
            self.decoders[f'decoder_{i}'] = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor, id_targets: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Shared encoder
        shared_features = self.shared_encoder(x)
        
        # Task-specific predictions
        logits = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            target_id = id_targets[i].item()
            decoder_name = f'decoder_{target_id}'
            if decoder_name in self.decoders:
                logits[i] = self.decoders[decoder_name](shared_features[i:i+1]).squeeze()
            else:
                # Fallback
                logits[i] = self.decoders['decoder_0'](shared_features[i:i+1]).squeeze()
        
        return logits


def create_model(model_type: str = 'multihead', **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'multihead' or 'multitask'
        **kwargs: model parameters
    
    Returns:
        model instance
    """
    if model_type == 'multihead':
        return MultiHeadMLP(**kwargs)
    elif model_type == 'multitask':
        return MultiTaskMLP(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    print("Testing MultiHeadMLP...")
    
    # Create model
    model = MultiHeadMLP(input_dim=100, hidden_dims=[256, 128, 64], num_heads=100)
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 100)
    id_targets = torch.randint(0, 100, (batch_size,))
    
    logits = model(x, id_targets)
    probs = model.predict_proba(x, id_targets)
    preds = model.predict(x, id_targets)
    
    print(f"Input shape: {x.shape}")
    print(f"ID targets shape: {id_targets.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}") 