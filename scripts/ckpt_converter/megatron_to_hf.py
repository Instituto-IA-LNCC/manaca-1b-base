#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Conversor Megatron (mcore) -> HuggingFace LlamaForCausalLM
=====================================================================

PT ------------------------------------------------------------------------
Converte o checkpoint mcore do Manaca-1B (Megatron-LM, fork llm-jp/nii-geniac)
para o formato HuggingFace, com DUAS correcoes que o saver_llama3_hf.py do fork
NAO faz e que sao essenciais para o nosso modelo:

  1. PRESERVA os biases. O modelo foi treinado com add_bias_linear=True, entao
     tem bias em todas as camadas lineares (qkv, proj, fc1, fc2). Medimos e eles
     sao significativos (|bias| da qkv > |peso|). O saver do fork so grava .weight
     e descartaria os biases. Aqui gravamos todos, com attention_bias=True e
     mlp_bias=True no config.
  2. IDS DE TOKEN corretos. O saver do fork chumba bos=128000/eos=128001 (Llama-3).
     O nosso tokenizador SentencePiece usa unk=0, bos=1, eos=2, pad=3 num vocab de
     64128. Usamos esses.

A logica de split do QKV com GQA e a MESMA do saver_llama3_hf.py do fork (por
grupo KV: [q_per, head_dim, head_dim]), entao a parte "dificil" segue validada.

SEGURANCA: le SOMENTE o checkpoint (nunca escreve nele) e grava o modelo HF num
diretorio NOVO. Roda em CPU (nao precisa de GPU).

Uso (dentro da imagem manaca-train, que tem torch + transformers):
    python megatron_to_hf.py \
        --load-dir /ckpt/manaca-1b \
        --save-dir /out/manaca-1b-hf \
        --tokenizer-model /tok/manaca-tokenizer.model \
        [--iter 20000] [--save-dtype bfloat16] [--validate]

EN ------------------------------------------------------------------------
Manaca-1B - Megatron (mcore) -> HuggingFace LlamaForCausalLM converter
Converts the Manaca-1B mcore checkpoint (Megatron-LM, llm-jp/nii-geniac fork) to
the HuggingFace format, with TWO fixes that the fork's saver_llama3_hf.py does
NOT do and that are essential for our model:

  1. PRESERVES the biases. The model was trained with add_bias_linear=True, so it
     has bias in every linear layer (qkv, proj, fc1, fc2). We measured them and they
     are significant (|bias| of qkv > |weight|). The fork's saver only writes .weight
     and would discard the biases. Here we write them all, with attention_bias=True
     and mlp_bias=True in the config.
  2. Correct TOKEN IDS. The fork's saver hardcodes bos=128000/eos=128001 (Llama-3).
     Our SentencePiece tokenizer uses unk=0, bos=1, eos=2, pad=3 in a vocab of
     64128. We use those.

The QKV-with-GQA split logic is the SAME as the fork's saver_llama3_hf.py (per
KV group: [q_per, head_dim, head_dim]), so the "hard" part remains validated.

SAFETY: it ONLY reads the checkpoint (never writes to it) and writes the HF model
to a NEW directory. Runs on CPU (no GPU needed).

Usage (inside the manaca-train image, which has torch + transformers):
    python megatron_to_hf.py \
        --load-dir /ckpt/manaca-1b \
        --save-dir /out/manaca-1b-hf \
        --tokenizer-model /tok/manaca-tokenizer.model \
        [--iter 20000] [--save-dtype bfloat16] [--validate]

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


def achar_checkpoint(load_dir: str, it: int | None) -> Path:
    """Aceita tanto .../manaca-1b (com iter_XXXX/ e latest_...txt) quanto .../iter_XXXX."""
    ld = Path(load_dir)
    marker = ld / "latest_checkpointed_iteration.txt"
    if it is None and marker.exists():
        it = int(marker.read_text().split()[0])
    if it is not None:
        cand = ld / f"iter_{it:07d}" / "mp_rank_00" / "model_optim_rng.pt"
        if cand.exists():
            return cand
    hits = sorted(glob.glob(str(ld / "**" / "model_optim_rng.pt"), recursive=True))
    if not hits:
        sys.exit(f"[ERRO] nao encontrei model_optim_rng.pt sob {ld}")
    return Path(hits[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Conversor mcore->HF do Manaca-1B (preserva biases)")
    ap.add_argument("--load-dir", required=True, help="Diretorio do checkpoint (a COPIA de trabalho).")
    ap.add_argument("--save-dir", required=True, help="Diretorio de SAIDA (novo, nunca por cima do original).")
    ap.add_argument("--tokenizer-model", required=True, help="Caminho do .model do SentencePiece.")
    ap.add_argument("--iter", type=int, default=None, help="Iteracao (default: le latest_checkpointed_iteration.txt).")
    ap.add_argument("--save-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--validate", action="store_true", help="Recarrega do disco e gera texto de sanidade.")
    args = ap.parse_args()

    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        sys.exit("[ERRO] transformers ausente - rode dentro da imagem manaca-train.")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.save_dtype]

    ckpt_file = achar_checkpoint(args.load_dir, args.iter)
    print(f"[conv] lendo {ckpt_file}")
    ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    sd = ck["model"]
    ma = ck["args"]

    H = ma.hidden_size
    nL = getattr(ma, "encoder_num_layers", None) or ma.num_layers
    nH = ma.num_attention_heads
    nKV = ma.num_query_groups
    V = ma.padded_vocab_size
    inter = ma.ffn_hidden_size
    hd = H // nH             # head_dim
    q_per = nH // nKV * hd   # colunas de Q por grupo KV
    # zero-centered gamma (weight-1) e uma pegadinha conhecida do Megatron.
    zc = bool(getattr(ma, "layernorm_zero_centered_gamma", False) or getattr(ma, "apply_layernorm_1p", False))
    off = 1.0 if zc else 0.0
    print(f"[conv] H={H} L={nL} heads={nH} kv={nKV} inter={inter} vocab={V} head_dim={hd} "
          f"attention_bias={getattr(ma,'add_bias_linear',None)} zero_centered_norm={zc}")

    def split_qkv(t: torch.Tensor, is_bias: bool):
        """Mesma logica do saver_llama3_hf.py: por grupo KV -> [q_per, hd, hd]."""
        if is_bias:
            t = t.view(nKV, -1)
            q, k, v = torch.split(t, [q_per, hd, hd], dim=1)
            return q.reshape(-1), k.reshape(-1), v.reshape(-1)
        t = t.view(nKV, -1, H)
        q, k, v = torch.split(t, [q_per, hd, hd], dim=1)
        return q.reshape(-1, H), k.reshape(-1, H), v.reshape(-1, H)

    new: dict[str, torch.Tensor] = {}

    def put(name: str, tensor: torch.Tensor):
        new[name] = tensor.to(dtype).contiguous()

    put("model.embed_tokens.weight", sd["embedding.word_embeddings.weight"])
    for L in range(nL):
        p = f"decoder.layers.{L}."
        hp = f"model.layers.{L}."
        # RMSNorm (fundidos nas camadas TE); +off trata zero-centered gamma se houver.
        put(hp + "input_layernorm.weight", sd[p + "self_attention.linear_qkv.layer_norm_weight"] + off)
        put(hp + "post_attention_layernorm.weight", sd[p + "mlp.linear_fc1.layer_norm_weight"] + off)
        # Atencao: QKV fundido -> q/k/v (peso E bias)
        qw, kw, vw = split_qkv(sd[p + "self_attention.linear_qkv.weight"], False)
        qb, kb, vb = split_qkv(sd[p + "self_attention.linear_qkv.bias"], True)
        put(hp + "self_attn.q_proj.weight", qw); put(hp + "self_attn.q_proj.bias", qb)
        put(hp + "self_attn.k_proj.weight", kw); put(hp + "self_attn.k_proj.bias", kb)
        put(hp + "self_attn.v_proj.weight", vw); put(hp + "self_attn.v_proj.bias", vb)
        put(hp + "self_attn.o_proj.weight", sd[p + "self_attention.linear_proj.weight"])
        put(hp + "self_attn.o_proj.bias", sd[p + "self_attention.linear_proj.bias"])
        # MLP SwiGLU: fc1 fundido -> gate(W, primeira metade) | up(V, segunda metade)
        fc1w = sd[p + "mlp.linear_fc1.weight"]
        fc1b = sd[p + "mlp.linear_fc1.bias"]
        put(hp + "mlp.gate_proj.weight", fc1w[:inter]); put(hp + "mlp.gate_proj.bias", fc1b[:inter])
        put(hp + "mlp.up_proj.weight", fc1w[inter:]);   put(hp + "mlp.up_proj.bias", fc1b[inter:])
        put(hp + "mlp.down_proj.weight", sd[p + "mlp.linear_fc2.weight"])
        put(hp + "mlp.down_proj.bias", sd[p + "mlp.linear_fc2.bias"])
    put("model.norm.weight", sd["decoder.final_layernorm.weight"] + off)
    put("lm_head.weight", sd["output_layer.weight"])

    del ck, sd

    cfg = LlamaConfig(
        vocab_size=V,
        hidden_size=H,
        intermediate_size=inter,
        num_hidden_layers=nL,
        num_attention_heads=nH,
        num_key_value_heads=nKV,
        max_position_embeddings=ma.max_position_embeddings,
        rms_norm_eps=ma.norm_epsilon,
        rope_theta=ma.rope_theta,
        attention_bias=True,
        mlp_bias=True,
        tie_word_embeddings=not ma.untie_embeddings_and_output_weights,
        bos_token_id=1, eos_token_id=2, pad_token_id=3,
        torch_dtype=args.save_dtype,
    )
    if not hasattr(cfg, "mlp_bias"):
        sys.exit("[ERRO] LlamaConfig sem mlp_bias; atualize transformers (>=4.41).")

    print("[conv] instanciando LlamaForCausalLM e carregando pesos (strict=True)...")
    model = LlamaForCausalLM(cfg)
    # Guarda-chuva: a versao do transformers PRECISA aplicar os biases, senao
    # eles seriam descartados em silencio (mesmo bug do saver do fork).
    if model.model.layers[0].mlp.gate_proj.bias is None:
        sys.exit("[ERRO] esta versao do transformers nao aplica mlp_bias no LlamaMLP; atualize (>=4.41).")
    if model.model.layers[0].self_attn.q_proj.bias is None:
        sys.exit("[ERRO] esta versao do transformers nao aplica attention_bias; atualize.")

    missing, unexpected = model.load_state_dict(new, strict=False)
    missing = [m for m in missing if not m.endswith("rotary_emb.inv_freq")]
    if missing:
        sys.exit(f"[ERRO] faltam chaves no state_dict (mapeamento incompleto): {missing[:10]}")
    if unexpected:
        sys.exit(f"[ERRO] chaves inesperadas (mapeamento errado): {unexpected[:10]}")
    del new

    model = model.to(dtype)
    os.makedirs(args.save_dir, exist_ok=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[conv] parametros: {n_par:,}  ({n_par/1e9:.3f} B)")
    print(f"[conv] salvando safetensors em {args.save_dir}")
    model.save_pretrained(args.save_dir, safe_serialization=True)

    # Tokenizador: envolve o SPM diretamente (ids batem com o treino).
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer(
        vocab_file=args.tokenizer_model,
        unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        add_bos_token=True, add_eos_token=False, legacy=False,
    )
    tok.save_pretrained(args.save_dir)
    print("[conv] tokenizador salvo.")
    # Notas finais bilingues (PT + EN) — o desfecho que o usuario le.
    print("[conv] NOTA: vocab do modelo (64128) > vocab do tokenizador (~64004); as linhas")
    print("[conv]       extras de embedding sao padding, nunca emitidas. Isso e normal.")
    print("[conv] NOTE: model vocab (64128) > tokenizer vocab (~64004); the extra")
    print("[conv]       embedding rows are padding, never emitted. This is normal.")
    print("[conv] OK.")

    if args.validate:
        validar(args.save_dir)
    return 0


def validar(save_dir: str):
    """Recarrega do disco e gera texto: se sair portugues coerente, os splits e biases batem."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\n[valida] recarregando do disco (AutoModelForCausalLM.from_pretrained)...")
    tok = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float32)
    model.eval()
    for prompt in ["A capital do Brasil e", "O Rio de Janeiro fica no", "A agua ferve a"]:
        ids = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=20, do_sample=False)
        print("  >", repr(tok.decode(out[0], skip_special_tokens=True)))
    print("[valida] se as frases acima fazem sentido em PT, a conversao esta fiel.")


if __name__ == "__main__":
    sys.exit(main())
