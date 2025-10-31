
def calculate_learnable_params(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    embeddings = d_model * vocab_size
    layers = num_layers * d_model * (4 * d_model + 3 * d_ff + 2)
    final = d_model * vocab_size + d_model
    return embeddings + layers + final

def calculate_matrix_multiplies_flops(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    # TB (bs sq, dmodel) -> (bs sq, dmodel)
    batch_size = 1
    seq_len = context_length  # ??
    d_k = d_model // num_heads
    norm = 2 * batch_size * seq_len * d_model
    mha_proj = 8 * batch_size * seq_len * d_model * d_model
    mha_rope = 4 * batch_size * seq_len * d_model * d_k
    mha_attn = 4 * batch_size * seq_len * seq_len * d_k
    ffn = 6 * d_model * d_ff
    final = 2 * batch_size * seq_len * vocab_size * d_model
    total = num_layers * (mha_proj + mha_rope + mha_attn + ffn + 2 * norm) + norm + final
    return total, mha_attn + mha_proj + mha_rope, ffn, final


def main():
    GPT_2_XL = {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 48,
        "d_model": 1600,
        "num_heads": 25,
        "d_ff": 6400,
    }
    GPT_2_SMALL = {
        "vocab_size": 50257,
        "context_length": 1024,
        "num_layers": 12,
        "d_model": 768,
        "num_heads": 12,
        "d_ff": 6400,
    }
    print("XL")
    t_p = calculate_learnable_params(**GPT_2_XL)
    print(f"Total number of trainable params: {t_p}")
    print(f"Memory {t_p * 4 / 1e9} GB")

    nbr_flops, mha, ffn, final = calculate_matrix_multiplies_flops(**GPT_2_XL)
    print(f"Number of G FLOPs, Total: {nbr_flops / 1e9}, MHA: {mha / 1e9}, SwiGLU: {ffn / 1e9}, Final: {final / 1e9}")

    print("small")
    t_p = calculate_learnable_params(**GPT_2_SMALL)
    print(f"Total number of trainable params: {t_p}")
    print(f"Memory {t_p * 4 / 1e9} GB")

    nbr_flops, mha, ffn, final = calculate_matrix_multiplies_flops(**GPT_2_SMALL)
    print(f"Number of G FLOPs, Total: {nbr_flops / 1e9}, MHA: {mha / 1e9}, SwiGLU: {ffn / 1e9}, Final: {final / 1e9}")


if __name__=='__main__':
    main()