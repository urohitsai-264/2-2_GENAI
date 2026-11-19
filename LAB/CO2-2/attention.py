
import torch
import torch.nn as nn
import math
import re
import hashlib

torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def display(title, matrix):
    print(f"\n{title}")
    print("-" * 70)
    print(matrix)


def get_embeddings(prompt, embed_dim=4):
    tokens = re.findall(r"\b[\w']+\b", prompt.lower())

    if not tokens:
        raise ValueError("Please enter a valid prompt.")

    if len(tokens) > 12:
        raise ValueError("Please enter at most 12 words.")

    vectors = []

    for word in tokens:
        digest = hashlib.sha256(word.encode()).digest()
        vector = [digest[i] / 255.0 for i in range(embed_dim)]
        vectors.append(vector)

    return tokens, torch.tensor(vectors, dtype=torch.float32)


# ---------------------------------------------------------
# SCALED DOT-PRODUCT ATTENTION
# ---------------------------------------------------------

def scaled_dot_product_attention(Q, K, V):
    print("\n" + "=" * 75)
    print("             SCALED DOT-PRODUCT ATTENTION")
    print("=" * 75)

    d_k = Q.shape[-1]

    display("QUERY MATRIX (Q)", Q)
    display("KEY MATRIX (K)", K)
    display("VALUE MATRIX (V)", V)

    # Step 1: Transpose K
    K_T = K.transpose(-2, -1)
    display("STEP 1: TRANSPOSE OF KEY MATRIX (K^T)", K_T)

    # Step 2: Matrix multiplication
    scores = torch.matmul(Q, K_T)
    display("STEP 2: ATTENTION SCORES (Q x K^T)", scores)

    # Step 3: Scaling
    scale = math.sqrt(d_k)

    print(f"\nd_k = {d_k}")
    print(f"sqrt(d_k) = {scale:.4f}")

    scaled_scores = scores / scale
    display("STEP 3: SCALED SCORE MATRIX", scaled_scores)

    # Step 4: Softmax
    weights = torch.softmax(scaled_scores, dim=-1)
    display("STEP 4: SOFTMAX ATTENTION WEIGHTS", weights)

    print("\nAttention weight row sums:")
    print(weights.sum(dim=-1))

    # Step 5: Final output
    output = torch.matmul(weights, V)
    display("STEP 5: FINAL ATTENTION OUTPUT (WEIGHTS x V)", output)

    return output


# ---------------------------------------------------------
# MULTI-HEAD ATTENTION
# ---------------------------------------------------------

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_length, embed_dim = x.shape

        print("\n" + "=" * 75)
        print("                    MULTI-HEAD ATTENTION")
        print("=" * 75)

        print("\nEmbedding Dimension:", embed_dim)
        print("Number of Heads:", self.num_heads)
        print("Dimension Per Head:", self.head_dim)

        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        display("MHA QUERY MATRIX (Q)", Q[0])
        display("MHA KEY MATRIX (K)", K[0])
        display("MHA VALUE MATRIX (V)", V[0])

        # Split into heads
        Q = Q.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.reshape(
            batch_size, seq_length,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        head_outputs = []

        # Calculate attention independently for each head
        for i in range(self.num_heads):

            head_no = i + 1

            print("\n" + "-" * 75)
            print(f"                    ATTENTION HEAD {head_no}")
            print("-" * 75)

            q = Q[0, i]
            k = K[0, i]
            v = V[0, i]

            display(f"HEAD {head_no} - QUERY MATRIX", q)
            display(f"HEAD {head_no} - KEY MATRIX", k)
            display(f"HEAD {head_no} - VALUE MATRIX", v)

            # Transpose
            k_t = k.transpose(-2, -1)
            display(f"HEAD {head_no} - K TRANSPOSE", k_t)

            # Scores
            scores = torch.matmul(q, k_t)
            display(f"HEAD {head_no} - ATTENTION SCORES", scores)

            # Scaling
            scaled = scores / math.sqrt(self.head_dim)
            display(f"HEAD {head_no} - SCALED SCORES", scaled)

            # Softmax
            weights = torch.softmax(scaled, dim=-1)
            display(f"HEAD {head_no} - ATTENTION WEIGHTS", weights)

            # Final head output
            output = torch.matmul(weights, v)
            display(f"HEAD {head_no} - FINAL OUTPUT", output)

            head_outputs.append(output)

        # Concatenate all heads
        concatenated = torch.cat(head_outputs, dim=-1)

        display("CONCATENATED HEAD OUTPUT", concatenated)

        # Final projection
        final_output = self.fc_out(concatenated)

        display("FINAL MULTI-HEAD ATTENTION OUTPUT", final_output[0])

        print("\nFinal output shape:", tuple(final_output.shape))

        return final_output


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():

    print("=" * 75)
    print("               WEEK 8 - GENERATIVE AI LAB")
    print("       SCALED DOT-PRODUCT AND MULTI-HEAD ATTENTION")
    print("=" * 75)

    prompt = input("\nEnter your text prompt: ").strip()

    if not prompt:
        print("Error: Prompt cannot be empty.")
        return

    try:
        tokens, X = get_embeddings(prompt)
    except ValueError as error:
        print("Error:", error)
        return

    print("\n1. INPUT PROCESSING")
    print("-" * 70)

    print("Original Prompt:", prompt)
    print("Tokens:", tokens)
    print("Number of Tokens:", len(tokens))

    display("INPUT EMBEDDING MATRIX (X)", X)
    print("Embedding Shape:", tuple(X.shape))

    # Use reproducible random projection matrices
    torch.manual_seed(42)

    embed_dim = X.shape[-1]

    W_Q = torch.randn(embed_dim, embed_dim) * 0.5
    W_K = torch.randn(embed_dim, embed_dim) * 0.5
    W_V = torch.randn(embed_dim, embed_dim) * 0.5

    print("\n2. GENERATING Q, K AND V")

    display("QUERY PROJECTION MATRIX (W_Q)", W_Q)
    display("KEY PROJECTION MATRIX (W_K)", W_K)
    display("VALUE PROJECTION MATRIX (W_V)", W_V)

    Q = torch.matmul(X, W_Q)
    K = torch.matmul(X, W_K)
    V = torch.matmul(X, W_V)

    # Scaled Dot-Product Attention
    scaled_dot_product_attention(Q, K, V)

    # Multi-Head Attention
    print("\n3. INITIALIZING MULTI-HEAD ATTENTION")

    torch.manual_seed(42)

    mha = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=2
    )

    with torch.no_grad():
        final_output = mha(X.unsqueeze(0))

    print("\n" + "=" * 75)
    print("                 EXECUTION COMPLETED")
    print("=" * 75)

    print("Scaled Dot-Product Attention: SUCCESS")
    print("Multi-Head Attention: SUCCESS")
    print("All matrix calculations displayed successfully.")


if __name__ == "__main__":
    main()
