from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "CS"
    def __init__(
        self,
        num_layers=12,
        padded_vocab_size=2414,
        hidden_size=768,
        ffn_hidden_size=768*4,
        kv_channels=48,
        num_attention_heads=12,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=True,
        bias_dropout_fusion=True,
        multi_query_attention=False,multi_query_group_num=2,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        original_rope = True,
        use_cache=True,
        activation_function="swiglu",
        torch_dtype="float16",
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
        num_labels=0,
        problem_type=None,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.original_rope = original_rope
        self.use_cache = use_cache
        self.activation_function = activation_function
        self.torch_dtype = torch_dtype
        self.tie_word_embeddings = tie_word_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.problem_type = problem_type
        
        super().__init__(**kwargs)