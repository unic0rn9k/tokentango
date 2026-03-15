from typing import Optional
import math
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx
from jaxtyping import Float, Int


EMBEDDING_DIM = 32


class PositionalEncoding(eqx.Module):
    value: Float[jax.Array, "max_seq_len embed_dim"]

    def __init__(self, max_seq_len: int, embedding_dim: int = EMBEDDING_DIM):
        positional_encoding = jnp.zeros((max_seq_len, embedding_dim))
        position = jnp.arange(0, max_seq_len).reshape(-1, 1).astype(jnp.float32)
        div_term = jnp.exp(
            jnp.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim)
        )
        positional_encoding = positional_encoding.at[:, 0::2].set(
            jnp.sin(position * div_term)
        )
        positional_encoding = positional_encoding.at[:, 1::2].set(
            jnp.cos(position * div_term)
        )
        self.value = positional_encoding.astype(jnp.float16)

    def __call__(
        self, x: Float[jax.Array, "batch seq embed"]
    ) -> Float[jax.Array, "batch seq embed"]:
        seq_len = x.shape[1]
        return x + self.value[:seq_len, :]


class TransformerEncoder(eqx.Module):
    """Placeholder transformer encoder - does nothing, implement later."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        *,
        key: jax.random.PRNGKey,
    ):
        pass

    def __call__(
        self, x: Float[jax.Array, "batch seq embed"]
    ) -> Float[jax.Array, "batch seq embed"]:
        return x


class BertClassifier(eqx.Module):
    embeddings: Float[jax.Array, "vocab_size embed_dim"]
    positional_encoding: PositionalEncoding
    transformer: TransformerEncoder
    preclassifier: eqx.nn.Linear
    classifier: eqx.nn.Linear
    mlm_head: eqx.nn.Linear

    vocab_size: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int = 6,
        num_heads: int = 4,
        embedding_dim: int = EMBEDDING_DIM,
        *,
        key: jax.random.PRNGKey,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        key_emb, key_trans, key_pre, key_cls, key_mlm = jax.random.split(key, 5)

        self.embeddings = (
            jax.random.normal(key_emb, (vocab_size, embedding_dim)).astype(jnp.float16)
            * 0.02
        )

        self.positional_encoding = PositionalEncoding(max_seq_len, embedding_dim)

        self.transformer = TransformerEncoder(
            embedding_dim, num_heads, num_layers, key=key_trans
        )

        self.preclassifier = eqx.nn.Linear(
            embedding_dim, embedding_dim, use_bias=True, key=key_pre
        )
        self.classifier = eqx.nn.Linear(embedding_dim, 1, use_bias=True, key=key_cls)
        self.mlm_head = eqx.nn.Linear(
            embedding_dim,
            vocab_size,
            use_bias=False,
            key=key_mlm,
        )

    def embed(
        self, seq: Int[jax.Array, "batch seq"]
    ) -> Float[jax.Array, "batch seq embed"]:
        x = jnp.take(self.embeddings, seq, axis=0)
        x = self.positional_encoding(x)
        return x

    def hidden(
        self, seq: Int[jax.Array, "batch seq"]
    ) -> Float[jax.Array, "batch seq embed"]:
        x = self.embed(seq)
        return self.transformer(x)

    def classify(
        self, hidden: Float[jax.Array, "batch seq embed"]
    ) -> Float[jax.Array, "batch 1"]:
        cls_hidden = hidden[:, 0, :]  # (batch, embed)
        x = jnn.relu(
            cls_hidden @ self.preclassifier.weight.T + self.preclassifier.bias
        )  # (batch, embed)
        return jnp.tanh(
            x @ self.classifier.weight.T + self.classifier.bias
        )  # (batch, 1)

    def classify_loss(
        self,
        hidden: Float[jax.Array, "batch seq embed"],
        mb_y: Float[jax.Array, "batch"],
    ) -> Float[jax.Array, ""]:
        logits = self.classify(hidden)
        diff = jnp.abs(logits.flatten() - mb_y)
        loss = jnp.where(
            diff <= 1.0,
            0.5 * diff**2,
            diff - 0.5,
        )
        return jnp.mean(loss)

    def mlm_loss(
        self,
        hidden: Float[jax.Array, "batch seq embed"],
        mb_x: Int[jax.Array, "batch seq"],
    ) -> Float[jax.Array, ""]:
        hidden_masked = hidden[:, 1:, :]  # (batch, seq-1, embed)
        batch_size, seq_len, embed_dim = hidden_masked.shape
        hidden_flat = hidden_masked.reshape(-1, embed_dim)  # (batch*(seq-1), embed)

        logits = hidden_flat @ self.mlm_head.weight.T  # (batch*(seq-1), vocab)
        targets = mb_x[:, 1:].reshape(-1)

        exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
        target_probs = probs[jnp.arange(len(targets)), targets]
        loss = -jnp.mean(jnp.log(target_probs + 1e-8))

        return loss
