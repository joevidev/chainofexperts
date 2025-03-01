from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

GPT_NEOX_DSMOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

class GPTNeoXDSMoEConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPTNeoXDSMoEModel`].
    It is used to instantiate a GPTNeoX model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoXDSMoEModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            Percentage of hidden dimensions to allocate to rotary embeddings.
        rotary_emb_base (`int`, *optional*, defaults to 10000):
            Base for rotary embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_parallel_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
            speedup at the cost of a slight quality degradation.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for MLP layers.
        expert_intermediate_size (`int`, *optional*, defaults to None):
            Dimension of the expert intermediate feedforward layers. 
            If not provided, intermediate_size will be used.
        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts. None means no shared experts.
        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts. None means no routed experts (standard dense model).
        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of experts to select per token. None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts' output.
        n_group (`int`, *optional*, defaults to None):
            Number of expert groups for group-based routing.
        topk_group (`int`, *optional*, defaults to None):
            Number of groups to select per token.
        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the topk expert weights.
        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Function to compute expert weights: 'softmax' or 'sigmoid'.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary load balancing loss weight.
        seq_aux (`bool`, *optional*, defaults to True):
            Whether to compute auxiliary loss for each sequence.
        attention_bias (`bool`, *optional*, defaults to False):
            Whether to use bias in attention linear layers.
        use_flash_attention (`bool`, *optional*, defaults to False):
            Whether to use flash attention for a potential speedup and reduced memory usage.
    """

    model_type = "gpt_neox_dsmoe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        attention_bias=True,
        # Backward compatibility params (original Pythia-MoE)
        moe_sparsity=1,
        moe_granularity=1,
        moe_topk=1,
        # DeepSeek MoE params
        expert_intermediate_size=None,
        n_shared_experts=None,
        n_routed_experts=None,
        num_experts_per_tok=None,
        routed_scaling_factor=1.0,
        n_group=None,
        topk_group=None,
        norm_topk_prob=False,
        scoring_func="softmax",
        aux_loss_alpha=0.001,
        seq_aux=True,
        use_flash_attention=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rope_theta = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias

        # Original Pythia-MoE parameters (for backward compatibility)
        self.moe_sparsity = moe_sparsity
        self.moe_granularity = moe_granularity
        self.moe_topk = moe_topk
        
        # DeepSeek MoE parameters
        self.expert_intermediate_size = expert_intermediate_size
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.attention_bias = attention_bias
        self.use_flash_attention = use_flash_attention
        
        super().__init__(**kwargs)