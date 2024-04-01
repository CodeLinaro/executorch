from typing import List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from executorch.examples.models.llama2.llama_transformer import (
    apply_rotary_emb,
    precompute_freqs_cis,
    FeedForward,
    ModelArgs,
    RMSNorm,
)

class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.n_heads
        self.head_dim = self.dim // self.num_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = self.num_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len

        self.wq = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        self.attn_softmax = torch.nn.Softmax(dim=-1)

        scale = float(self.head_dim) ** -0.5
        scale_tensor = torch.tensor(
            [scale], dtype=torch.float32, requires_grad=False
        ).view(1, 1, 1)
        self.register_buffer("scale_tensor", scale_tensor, False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = hidden_states.shape

        # QKV
        q, k, v = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        # We need view_copy elimination
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # Use the incoming cache
        k_cache = k_cache.view(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim)
        v_cache = v_cache.view(bsz, self.max_seq_len, self.n_kv_heads, self.head_dim)
        k = k_cache * (1.0 - k_mask) + k * k_mask
        v = v_cache * (1.0 - v_mask) + v * v_mask
        q = q.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.transpose(-2, -1)
        attn = q @ k
        attn = attn * self.scale_tensor
        attn += attention_mask
        attn = self.attn_softmax(attn)
        y = attn @ v
        y = y.transpose(0, 1).contiguous().view(bsz, seqlen, self.dim)
        y = self.wo(y)

        # restore kv cache
        k = k.transpose(-2, -1).transpose(1, 2)
        v = v.transpose(1, 2)
        return y, k, v


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.attention = LlamaAttention(config=config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # Self Attention
        hidden_states, k_cache, v_cache = self.attention(
            hidden_states=self.attention_norm(x),
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            attention_mask=attention_mask,
            k_cache=k_cache,
            v_cache=v_cache,
            k_mask=k_mask,
            v_mask=v_mask,
        )
        hidden_states = x + hidden_states
        output = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return output, k_cache, v_cache


class LlamaModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_freq_base,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        output_k_cache = []
        output_v_cache = []
        freqs_cos = self.freqs_cos[input_pos]
        freqs_sin = self.freqs_sin[input_pos]

        hidden_states = self.tok_embeddings(tokens)
        for ind, decoder_layer in enumerate(self.layers):
            k = k_cache[ind, :, :, :]
            v = v_cache[ind, :, :, :]
            hidden_states, k, v = decoder_layer(
                hidden_states,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                attention_mask=attention_mask,
                k_cache=k,
                v_cache=v,
                k_mask=k_mask,
                v_mask=v_mask,
            )
            output_k_cache.append(
                k.view(self.config.max_batch_size, self.config.max_seq_len, self.config.dim)
            )
            output_v_cache.append(
                v.view(self.config.max_batch_size, self.config.max_seq_len, self.config.dim)
            )

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        # update kv cache
        output_k_cache = torch.concat(output_k_cache)
        output_k_cache = output_k_cache.view(
            self.config.n_layers, self.config.max_batch_size, self.config.max_seq_len, self.config.dim
        )
        output_v_cache = torch.concat(output_v_cache)
        output_v_cache = output_v_cache.view(
            self.config.n_layers, self.config.max_batch_size, self.config.max_seq_len, self.config.dim
        )
        return logits, output_k_cache, output_v_cache

    def get_example_inputs(self):
        tokens = torch.tensor([[1]], dtype=torch.int32)
        pos_ids = torch.tensor([0], dtype=torch.int32)
        head_dim = self.config.dim // self.config.n_heads
        attention_mask = torch.full(
            (1, 1, 1, self.config.max_seq_len),
            -10000,
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)
        k_mask = torch.zeros(
            self.config.max_batch_size,
            self.config.max_seq_len,
            self.config.n_heads,
            head_dim
        )
        k_mask[:, 0, :, :] = torch.ones(
            self.config.max_batch_size,
            self.config.n_heads,
            head_dim
        )
        v_mask = torch.zeros(
            self.config.max_batch_size,
            self.config.max_seq_len,
            self.config.n_heads,
            head_dim
        )
        v_mask[:, 0, :, :] = torch.ones(
            self.config.max_batch_size,
            self.config.n_heads,
            head_dim
        )
        k_cache = torch.zeros(
            self.config.n_layers,
            self.config.max_batch_size,
            self.config.max_seq_len,
            self.config.dim,
        )
        v_cache = torch.zeros(
            self.config.n_layers,
            self.config.max_batch_size,
            self.config.max_seq_len,
            self.config.dim,
        )
        return (
            tokens,
            pos_ids,
            attention_mask,
            k_cache,
            v_cache,
            k_mask,
            v_mask,
        )


from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import (
    QcomChipset,
)
from executorch.backends.qualcomm.utils.utils import (
    capture_program,
    canonicalize_program,
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.program._program import ExirExportedProgram, EdgeCompileConfig


class CompositeLlama(torch.nn.Module):
    def __init__(self, division, config) -> None:
        super().__init__()
        self.llama = LlamaModel(config).eval()
        inputs = self.llama.get_example_inputs()

        # backend option
        backend_options = generate_htp_compiler_spec(
            use_fp16=True, use_multi_contexts=True
        )
        compiler_specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SM8650,
            backend_options=backend_options,
        )
        partitioner = QnnPartitioner(compiler_specs)

        self.division, self.lowered_modules = division, []
        # embedding
        edge_prog = capture_program(self.llama.tok_embeddings, (inputs[0],))
        edge_prog.exported_program = to_backend(
            edge_prog.exported_program, partitioner
        )
        self.lowered_modules.append(
            edge_prog.exported_program.graph_module._modules.get("lowered_module_0")
        )

        # attentions
        def get_block_module(llama, indexes):
            class LlamaBlock(torch.nn.Module):
                def __init__(self, llama, indexes) -> None:
                    super().__init__()
                    self.llama = llama
                    self.indexes = indexes
                    self.config = llama.config

                def forward(
                    self,
                    hidden_states,
                    freqs_cos,
                    freqs_sin,
                    attention_mask,
                    k_cache,
                    v_cache,
                    k_mask,
                    v_mask,
                ):
                    output_k_cache, output_v_cache = [], []
                    for ind in self.indexes:
                        k = k_cache[ind, :, :, :]
                        v = v_cache[ind, :, :, :]
                        hidden_states, k, v = self.llama.layers[ind](
                            hidden_states,
                            freqs_cos=freqs_cos,
                            freqs_sin=freqs_sin,
                            attention_mask=attention_mask,
                            k_cache=k,
                            v_cache=v,
                            k_mask=k_mask,
                            v_mask=v_mask,
                        )
                        output_k_cache.append(
                            k.view(self.config.max_batch_size, self.config.max_seq_len, self.config.dim)
                        )
                        output_v_cache.append(
                            v.view(self.config.max_batch_size, self.config.max_seq_len, self.config.dim)
                        )

                    return hidden_states, output_k_cache, output_v_cache

            return LlamaBlock(llama, indexes)

        layers_per_ctx = config.n_layers // division
        self.input_indexes = []
        for i in range(division):
            indexes = [*range(layers_per_ctx*i, layers_per_ctx*(i+1))]
            self.input_indexes.append(indexes)
            llama_block = get_block_module(self.llama, indexes)
            with torch.no_grad():
                edge_prog = capture_program(llama_block, (
                        self.llama.tok_embeddings(inputs[0]),
                        self.llama._buffers['freqs_cos'][inputs[1]],
                        self.llama._buffers['freqs_sin'][inputs[1]],
                        *inputs[2:],
                    )
                )
                edge_prog.exported_program = to_backend(
                    edge_prog.exported_program, partitioner
                )
                self.lowered_modules.append(
                    edge_prog.exported_program.graph_module._modules.get("lowered_module_0")
                )

        # affine layer
        def get_affine_module(llama):
            class LlamaAffine(torch.nn.Module):
                def __init__(self, llama) -> None:
                        super().__init__()
                        self.llama = llama

                def forward(self, hidden_states):
                    hidden_states = self.llama.norm(hidden_states)
                    logits = self.llama.output(hidden_states)
                    return logits

            return LlamaAffine(llama)

        affine_block = get_affine_module(self.llama)
        with torch.no_grad():
            edge_prog = capture_program(affine_block, (
                self.llama.tok_embeddings(inputs[0]),)
            )
            edge_prog.exported_program = to_backend(
                edge_prog.exported_program, partitioner
            )
            self.lowered_modules.append(
                edge_prog.exported_program.graph_module._modules.get("lowered_module_0")
            )

    def forward(
        self,
        tokens: torch.Tensor,
        input_pos: torch.Tensor,
        attention_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ):
        output_k_cache, output_v_cache = [], []
        freqs_cos = self.llama._buffers['freqs_cos'][input_pos]
        freqs_sin = self.llama._buffers['freqs_sin'][input_pos]

        hidden_states = self.lowered_modules[0](tokens)[0]
        for i in range(self.division):
            hidden_states, okc, ovc = self.lowered_modules[i+1](
                hidden_states,
                freqs_cos,
                freqs_sin,
                attention_mask,
                k_cache,
                v_cache,
                k_mask,
                v_mask,
            )
            output_k_cache.append(*okc)
            output_v_cache.append(*ovc)

        logits = self.lowered_modules[-1](hidden_states)[0]
        output_k_cache = torch.concat(output_k_cache)
        output_v_cache = torch.concat(output_v_cache)
        output_k_cache = output_k_cache.view(
            self.config.n_layers, self.config.max_batch_size, self.config.max_seq_len, self.config.dim
        )
        output_v_cache = output_v_cache.view(
            self.config.n_layers, self.config.max_batch_size, self.config.max_seq_len, self.config.dim
        )
        # TODO: update k_mask / v_mask which could be handled by CPU

        return logits, output_k_cache, output_v_cache

    def get_example_input(self):
        return self.llama.get_example_inputs()

if __name__ == "__main__":
    config = ModelArgs()
    config.vocab_size = 512
    config.max_batch_size = 1
    config.max_seq_len = 128
    config.n_layers = 32
    config.dim = 128

    # plain export
    """ module = LlamaModel(config)
    inputs = module.get_example_inputs()

    logits, k, v = module(*inputs)

    prog = capture_program(module, inputs)
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8650,
        backend_options=backend_options,
    )
    qnn_partitioner = QnnPartitioner(compiler_specs)
    prog.exported_program = to_backend(prog.exported_program, qnn_partitioner)
    exec_prog = prog.to_executorch() """

    # composite scenario
    module = CompositeLlama(division=4, config=config).eval()
    sample_input = module.get_example_input()
    edge_prog = ExirExportedProgram(
        torch.export.export(module, sample_input),
        after_to_edge_passes=False,
    ).to_edge(EdgeCompileConfig(_check_ir_validity=False))
    canonicalize_program(edge_prog.exported_program)
    exec_prog = edge_prog.to_executorch()
