import torch
import torch.nn as nn
import torch.nn.functional as F

class BiShareAdapter(nn.Module):
    """
    A module that implements a bidirectional shared adapter with multi-head attention.
    
    Attributes:
        hidden_dim (int): The dimension of the hidden layer.
        num_heads (int): The number of heads in multi-head attention.
        l1 (nn.Linear): The first linear transformation.
        l2 (nn.Linear): The second linear transformation that projects back to the hidden_dim.
        multihead_attention1 (nn.MultiheadAttention): The multi-head attention layer.
        gate1 (nn.Parameter): A learnable gate parameter for blending attention output with input.

    """
    def __init__(self, hidden_dim, num_heads):
        """
        Inits BiShareAdapter with hidden dimension and number of attention heads.
        
        Args:
            hidden_dim (int): The dimension of the hidden layer.
            num_heads (int): The number of heads in multi-head attention.
        """
        super(BiShareAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.l1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.l2 = nn.Linear(hidden_dim//2, hidden_dim)

        # Add multi-head attention
        self.multihead_attention1 = nn.MultiheadAttention(hidden_dim//2, num_heads)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)

        self.init_weights()
        
    def init_weights(self):
        """Initializes weights with zeros."""
        self.l2.weight.data.zero_()
        self.l2.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the BiShareAdapter.
        
        Args:
            x (Tensor): The input tensor.
            
        Returns:
            Tensor: The output tensor after the adapter processing.
        """
        xinit = x
        x = self.l1(x)
        x2 = x
        attn_output, _ = self.multihead_attention1(x, x, x)
        x = F.gelu(x)
        alpha = torch.sigmoid(self.gate1)
        attn = alpha * attn_output + (1 - alpha) * x2
        x = self.l2(attn)

        return x + xinit

    
class MMadapter(nn.Module):
    """
    A module that implements a multimodal adapter with shared components and multi-head attention.
    
    Attributes:
        img_proj_down (nn.Linear): Linear layer that projects the input to a lower dimension.
        img_proj_up (nn.Linear): Linear layer that projects back to the original dimension.
        BiShareAdapterxx (BiShareAdapter or None): Optional shared BiShareAdapter instance.
        multihead_attention (nn.MultiheadAttention): Multi-head attention layer.
        gate1 (nn.Parameter): A learnable gate parameter for blending attention output with input.
    """
    def __init__(self, share_adapter, hidden_size, layer_id=0):
        """
        Inits MMadapter with a shared adapter, hidden size, and layer ID.
        
        Args:
            share_adapter (BiShareAdapter or None): The shared BiShareAdapter instance, if any.
            hidden_size (int): The size of the hidden layer.
            layer_id (int, optional): The layer ID, defaults to 0.
        """
        super(MMadapter, self).__init__()
        self.img_proj_down = nn.Linear(hidden_size, 128)
        self.img_proj_up = nn.Linear(128, hidden_size)
        self.BiShareAdapterxx = share_adapter
        self.multihead_attention = nn.MultiheadAttention(128, 8)
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        """Initializes weights with zeros."""
        self.img_proj_up.weight.data.zero_()
        self.img_proj_up.bias.data.zero_()

    def forward(self, x):
        """
        Forward pass of the MMadapter.
        
        Args:
            x (Tensor): The input tensor.
            
        Returns:
            Tensor: The output tensor after the adapter processing.
        """
        x_init = x
        x = self.img_proj_down(x)
        x = F.gelu(x)
        xmid = x
        x, _ = self.multihead_attention(x, x, x)
        if self.BiShareAdapterxx is not None:
            x = self.BiShareAdapterxx(x)
        x, _ = self.multihead_attention(x, x, x)
        alpha = torch.sigmoid(self.gate1)
        x = alpha * xmid + (1 - alpha) * x
        x = self.img_proj_up(x)
        x = x_init + x

        return x
