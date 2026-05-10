use crate::model_config::{ArchitectureSpec, TextLayerKind};

/// RoPE frequency scaling strategy.
///
/// - `Default`: Standard RoPE with optional linear scaling via `rope_scaling_factor`.
/// - `Linear`: Explicit linear interpolation (same math as Default but self-documenting).
/// - `Llama3`: LLaMA3 extended-context RoPE with per-dimension smooth blending.
/// - `Yarn`: YaRN extended-context RoPE (same blending, different parameter names, plus attention scale).
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingMode {
    Default,
    Linear {
        factor: f32,
    },
    Llama3 {
        factor: f32,
        original_max_pos: usize,
        low_freq_factor: f32,
        high_freq_factor: f32,
    },
    Yarn {
        factor: f32,
        original_max_pos: usize,
        beta_fast: f32,
        beta_slow: f32,
        /// Multiplier applied to attention logits (1/sqrt(d_head) * attention_factor).
        attention_factor: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_scaling_factor: f32,
    pub rope_mode: RopeScalingMode,
    pub attention_head_dim: usize,
    pub partial_rotary_factor: f32,
    pub qk_norm: bool,
    pub zero_centered_rmsnorm: bool,
    pub gated_query_attention: bool,
    pub tie_word_embeddings: bool,
    pub linear_attention: Option<LinearAttentionConfig>,
    /// Sliding window attention size (Mistral-style). `None` = full attention.
    pub sliding_window: Option<usize>,
    /// Mixture-of-Experts config.  `None` for dense models.
    pub moe: Option<MoeConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionConfig {
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
}

impl LinearAttentionConfig {
    pub fn key_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }
    pub fn value_dim(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

/// Configuration for Mixture-of-Experts FFN layers.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeConfig {
    /// Total number of experts in each MoE layer.
    pub n_experts: usize,
    /// Number of experts activated per token (top-k).
    pub n_experts_per_tok: usize,
    /// Hidden dimension inside each expert (may differ from the dense `hidden_dim`).
    pub moe_hidden_dim: usize,
    /// Number of shared (always-active) experts (0 for Mixtral, 1 for Qwen2-MoE).
    pub n_shared_experts: usize,
}

/// Weight tensor names for one MoE expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeExpertWeights {
    /// Gate / w1 projection: [moe_hidden × dim].
    pub gate_proj: String,
    /// Up   / w3 projection: [moe_hidden × dim].
    pub up_proj: String,
    /// Down / w2 projection: [dim × moe_hidden].
    pub down_proj: String,
}

/// Weight names for a sparse-MoE FFN sub-layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeMlpWeights {
    /// Router gate weight: [n_experts × dim].  Kept as F16 for routing precision.
    pub gate: String,
    /// Indexed by expert id.
    pub experts: Vec<MoeExpertWeights>,
    /// Optional always-active shared expert (Qwen2/3-MoE style).
    pub shared_expert: Option<MoeExpertWeights>,
}

pub struct TransformerWeights {
    pub token_embedding: String,
    pub layer_weights: Vec<LayerWeights>,
    pub norm_final: String,
    pub output: String,
}

pub struct LayerWeights {
    pub attention_wq: String,
    pub attention_wk: String,
    pub attention_wv: String,
    pub attention_wo: String,
    pub feed_forward_w1: String,
    pub feed_forward_w2: String,
    pub feed_forward_w3: String,
    pub attention_norm: String,
    pub ffn_norm: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybridAttentionWeights {
    FullAttention {
        q_proj: String,
        k_proj: String,
        v_proj: String,
        o_proj: String,
        q_norm: Option<String>,
        k_norm: Option<String>,
    },
    LinearAttention {
        in_proj_qkv: String,
        in_proj_z: String,
        in_proj_b: String,
        in_proj_a: String,
        out_proj: String,
        conv1d_weight: String,
        conv1d_bias: Option<String>,
        norm: String,
        a_log: String,
        dt_bias: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridLayerWeights {
    pub attention: HybridAttentionWeights,
    pub feed_forward_w1: String,
    pub feed_forward_w2: String,
    pub feed_forward_w3: String,
    pub attention_norm: String,
    pub ffn_norm: String,
    /// Set when this layer uses sparse MoE FFN instead of dense FFN.
    /// When `Some`, `feed_forward_w1/w2/w3` should be treated as unused.
    pub moe: Option<MoeMlpWeights>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridTransformerWeights {
    pub token_embedding: String,
    pub norm_final: String,
    pub output: String,
    pub layer_weights: Vec<HybridLayerWeights>,
    pub mtp_fc: Option<String>,
    pub has_vision_branch: bool,
}

impl TransformerWeights {
    pub fn from_tensor_names(names: &[&str]) -> Self {
        let has_name = |needle: &str| names.iter().any(|n| *n == needle);

        let hf_prefix = if names
            .iter()
            .any(|n| n.starts_with("model.language_model.layers."))
        {
            Some(("model.language_model.layers", 3usize))
        } else if names.iter().any(|n| n.starts_with("model.layers.")) {
            Some(("model.layers", 2usize))
        } else if names
            .iter()
            .any(|n| n.starts_with("layers.") && n.contains(".self_attn."))
        {
            Some(("layers", 1usize))
        } else {
            None
        };

        if let Some((layer_prefix, layer_idx_pos)) = hf_prefix {
            let mut n_layers = 0;
            let layer_search = format!("{layer_prefix}.");
            for &name in names {
                if name.starts_with(&layer_search) {
                    let parts: Vec<&str> = name.split('.').collect();
                    if let Some(idx_str) = parts.get(layer_idx_pos) {
                        if let Ok(l) = idx_str.parse::<usize>() {
                            n_layers = n_layers.max(l + 1);
                        }
                    }
                }
            }

            let mut layer_weights = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                layer_weights.push(LayerWeights {
                    attention_wq: format!("{layer_prefix}.{i}.self_attn.q_proj.weight"),
                    attention_wk: format!("{layer_prefix}.{i}.self_attn.k_proj.weight"),
                    attention_wv: format!("{layer_prefix}.{i}.self_attn.v_proj.weight"),
                    attention_wo: format!("{layer_prefix}.{i}.self_attn.o_proj.weight"),
                    feed_forward_w1: format!("{layer_prefix}.{i}.mlp.gate_proj.weight"),
                    feed_forward_w2: format!("{layer_prefix}.{i}.mlp.down_proj.weight"),
                    feed_forward_w3: format!("{layer_prefix}.{i}.mlp.up_proj.weight"),
                    attention_norm: format!("{layer_prefix}.{i}.input_layernorm.weight"),
                    ffn_norm: format!("{layer_prefix}.{i}.post_attention_layernorm.weight"),
                });
            }

            let token_embedding = if has_name("model.language_model.embed_tokens.weight") {
                "model.language_model.embed_tokens.weight"
            } else if has_name("model.embed_tokens.weight") {
                "model.embed_tokens.weight"
            } else {
                "tok_embeddings.weight"
            };

            let norm_final = if has_name("model.language_model.norm.weight") {
                "model.language_model.norm.weight"
            } else if has_name("model.norm.weight") {
                "model.norm.weight"
            } else {
                "norm.weight"
            };

            let output = if has_name("lm_head.weight") {
                "lm_head.weight"
            } else if has_name("model.language_model.lm_head.weight") {
                "model.language_model.lm_head.weight"
            } else if token_embedding == "model.language_model.embed_tokens.weight"
                || token_embedding == "model.embed_tokens.weight"
                || token_embedding == "tok_embeddings.weight"
            {
                token_embedding
            } else {
                "output.weight"
            };

            return Self {
                token_embedding: token_embedding.to_string(),
                norm_final: norm_final.to_string(),
                output: output.to_string(),
                layer_weights,
            };
        }

        let mut n_layers = 0;
        for &name in names {
            if name.starts_with("layers.") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Some(idx_str) = parts.get(1) {
                    if let Ok(l) = idx_str.parse::<usize>() {
                        n_layers = n_layers.max(l + 1);
                    }
                }
            }
        }

        let mut layer_weights = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layer_weights.push(LayerWeights {
                attention_wq: format!("layers.{i}.attention.wq.weight"),
                attention_wk: format!("layers.{i}.attention.wk.weight"),
                attention_wv: format!("layers.{i}.attention.wv.weight"),
                attention_wo: format!("layers.{i}.attention.wo.weight"),
                feed_forward_w1: format!("layers.{i}.feed_forward.w1.weight"),
                feed_forward_w2: format!("layers.{i}.feed_forward.w2.weight"),
                feed_forward_w3: format!("layers.{i}.feed_forward.w3.weight"),
                attention_norm: format!("layers.{i}.attention_norm.weight"),
                ffn_norm: format!("layers.{i}.ffn_norm.weight"),
            });
        }

        Self {
            token_embedding: "tok_embeddings.weight".to_string(),
            norm_final: "norm.weight".to_string(),
            output: "output.weight".to_string(),
            layer_weights,
        }
    }

    pub fn from_llama_style(model: &lns_core::format::ArchivedLnsModel) -> Self {
        let names: Vec<&str> = model.tensors.iter().map(|t| t.name.as_str()).collect();
        Self::from_tensor_names(&names)
    }
}

impl HybridTransformerWeights {
    pub fn from_tensor_names(names: &[&str], spec: &ArchitectureSpec) -> Self {
        let has_name = |needle: &str| names.iter().any(|n| *n == needle);
        let layer_prefix = if names
            .iter()
            .any(|n| n.starts_with("model.language_model.layers."))
        {
            "model.language_model.layers"
        } else if names.iter().any(|n| n.starts_with("model.layers.")) {
            "model.layers"
        } else {
            "layers"
        };

        let token_embedding = if has_name("model.language_model.embed_tokens.weight") {
            "model.language_model.embed_tokens.weight"
        } else if has_name("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else {
            "tok_embeddings.weight"
        };

        let norm_final = if has_name("model.language_model.norm.weight") {
            "model.language_model.norm.weight"
        } else if has_name("model.norm.weight") {
            "model.norm.weight"
        } else {
            "norm.weight"
        };

        let output = if has_name("lm_head.weight") {
            "lm_head.weight"
        } else if has_name("model.language_model.lm_head.weight") {
            "model.language_model.lm_head.weight"
        } else if token_embedding == "model.language_model.embed_tokens.weight"
            || token_embedding == "model.embed_tokens.weight"
            || token_embedding == "tok_embeddings.weight"
        {
            token_embedding
        } else {
            "output.weight"
        };

        let mut layer_weights = Vec::with_capacity(spec.text_layers.len());

        // ── MoE detection ─────────────────────────────────────────────────────
        // Mixtral-style:  {prefix}.0.block_sparse_moe.gate.weight
        //                 {prefix}.0.block_sparse_moe.experts.{j}.w1/w2/w3.weight
        // Qwen2/3-MoE:    {prefix}.0.mlp.gate.weight
        //                 {prefix}.0.mlp.experts.{j}.gate_proj/up_proj/down_proj.weight
        //                 {prefix}.0.mlp.shared_expert.gate_proj.weight  (optional)
        let is_block_sparse_moe =
            has_name(&format!("{layer_prefix}.0.block_sparse_moe.gate.weight"));
        let is_mlp_moe = !is_block_sparse_moe
            && has_name(&format!("{layer_prefix}.0.mlp.gate.weight"))
            && has_name(&format!("{layer_prefix}.0.mlp.experts.0.gate_proj.weight"));
        let is_moe = is_block_sparse_moe || is_mlp_moe;

        // Count experts by scanning for max index in layer 0.
        let n_moe_experts = if is_moe {
            let mut max_j = 0usize;
            for &name in names {
                let seg = if is_block_sparse_moe {
                    ".block_sparse_moe.experts."
                } else {
                    ".mlp.experts."
                };
                let needle = format!("{layer_prefix}.0{seg}");
                if let Some(rest) = name.strip_prefix(&needle) {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(j) = rest[..dot].parse::<usize>() {
                            max_j = max_j.max(j + 1);
                        }
                    }
                }
            }
            max_j
        } else {
            0
        };

        for (index, layer_kind) in spec.text_layers.iter().enumerate() {
            let attention = match layer_kind {
                TextLayerKind::FullAttention
                | TextLayerKind::MlpOnly
                | TextLayerKind::Unknown(_) => HybridAttentionWeights::FullAttention {
                    q_proj: format!("{layer_prefix}.{index}.self_attn.q_proj.weight"),
                    k_proj: format!("{layer_prefix}.{index}.self_attn.k_proj.weight"),
                    v_proj: format!("{layer_prefix}.{index}.self_attn.v_proj.weight"),
                    o_proj: format!("{layer_prefix}.{index}.self_attn.o_proj.weight"),
                    q_norm: has_name(&format!("{layer_prefix}.{index}.self_attn.q_norm.weight"))
                        .then(|| format!("{layer_prefix}.{index}.self_attn.q_norm.weight")),
                    k_norm: has_name(&format!("{layer_prefix}.{index}.self_attn.k_norm.weight"))
                        .then(|| format!("{layer_prefix}.{index}.self_attn.k_norm.weight")),
                },
                TextLayerKind::LinearAttention => HybridAttentionWeights::LinearAttention {
                    in_proj_qkv: format!("{layer_prefix}.{index}.linear_attn.in_proj_qkv.weight"),
                    in_proj_z: format!("{layer_prefix}.{index}.linear_attn.in_proj_z.weight"),
                    in_proj_b: format!("{layer_prefix}.{index}.linear_attn.in_proj_b.weight"),
                    in_proj_a: format!("{layer_prefix}.{index}.linear_attn.in_proj_a.weight"),
                    out_proj: format!("{layer_prefix}.{index}.linear_attn.out_proj.weight"),
                    conv1d_weight: format!("{layer_prefix}.{index}.linear_attn.conv1d.weight"),
                    conv1d_bias: has_name(&format!(
                        "{layer_prefix}.{index}.linear_attn.conv1d.bias"
                    ))
                    .then(|| format!("{layer_prefix}.{index}.linear_attn.conv1d.bias")),
                    norm: format!("{layer_prefix}.{index}.linear_attn.norm.weight"),
                    a_log: format!("{layer_prefix}.{index}.linear_attn.A_log"),
                    dt_bias: format!("{layer_prefix}.{index}.linear_attn.dt_bias"),
                },
            };

            layer_weights.push(HybridLayerWeights {
                attention,
                feed_forward_w1: format!("{layer_prefix}.{index}.mlp.gate_proj.weight"),
                feed_forward_w2: format!("{layer_prefix}.{index}.mlp.down_proj.weight"),
                feed_forward_w3: format!("{layer_prefix}.{index}.mlp.up_proj.weight"),
                attention_norm: format!("{layer_prefix}.{index}.input_layernorm.weight"),
                ffn_norm: format!("{layer_prefix}.{index}.post_attention_layernorm.weight"),
                moe: if is_moe {
                    let (gate, expert_names) = if is_block_sparse_moe {
                        let g = format!("{layer_prefix}.{index}.block_sparse_moe.gate.weight");
                        let experts = (0..n_moe_experts)
                            .map(|j| MoeExpertWeights {
                                gate_proj: format!(
                                    "{layer_prefix}.{index}.block_sparse_moe.experts.{j}.w1.weight"
                                ),
                                up_proj: format!(
                                    "{layer_prefix}.{index}.block_sparse_moe.experts.{j}.w3.weight"
                                ),
                                down_proj: format!(
                                    "{layer_prefix}.{index}.block_sparse_moe.experts.{j}.w2.weight"
                                ),
                            })
                            .collect();
                        (g, experts)
                    } else {
                        let g = format!("{layer_prefix}.{index}.mlp.gate.weight");
                        let experts = (0..n_moe_experts)
                            .map(|j| MoeExpertWeights {
                                gate_proj: format!(
                                    "{layer_prefix}.{index}.mlp.experts.{j}.gate_proj.weight"
                                ),
                                up_proj: format!(
                                    "{layer_prefix}.{index}.mlp.experts.{j}.up_proj.weight"
                                ),
                                down_proj: format!(
                                    "{layer_prefix}.{index}.mlp.experts.{j}.down_proj.weight"
                                ),
                            })
                            .collect();
                        (g, experts)
                    };
                    let shared_expert = if is_mlp_moe
                        && has_name(&format!(
                            "{layer_prefix}.{index}.mlp.shared_expert.gate_proj.weight"
                        )) {
                        Some(MoeExpertWeights {
                            gate_proj: format!(
                                "{layer_prefix}.{index}.mlp.shared_expert.gate_proj.weight"
                            ),
                            up_proj: format!(
                                "{layer_prefix}.{index}.mlp.shared_expert.up_proj.weight"
                            ),
                            down_proj: format!(
                                "{layer_prefix}.{index}.mlp.shared_expert.down_proj.weight"
                            ),
                        })
                    } else {
                        None
                    };
                    Some(MoeMlpWeights {
                        gate,
                        experts: expert_names,
                        shared_expert,
                    })
                } else {
                    None
                },
            });
        }

        Self {
            token_embedding: token_embedding.to_string(),
            norm_final: norm_final.to_string(),
            output: output.to_string(),
            layer_weights,
            mtp_fc: has_name("mtp.fc.weight").then(|| "mtp.fc.weight".to_string()),
            has_vision_branch: spec.has_vision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HybridAttentionWeights, HybridTransformerWeights, TransformerWeights};
    use crate::model_config::{ArchitectureSpec, TextLayerKind};

    #[test]
    fn maps_llama3_hf_names() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];
        let weights = TransformerWeights::from_tensor_names(&names);
        assert_eq!(weights.token_embedding, "model.embed_tokens.weight");
        assert_eq!(weights.norm_final, "model.norm.weight");
        assert_eq!(weights.output, "lm_head.weight");
        assert_eq!(weights.layer_weights.len(), 1);
    }

    #[test]
    fn maps_mistral_hf_names() {
        let names = vec![
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
        ];
        let weights = TransformerWeights::from_tensor_names(&names);
        assert_eq!(weights.layer_weights.len(), 2);
        assert_eq!(
            weights.layer_weights[1].attention_wq,
            "model.layers.1.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn maps_qwen3_text_names() {
        let names = vec![
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
        ];
        let weights = TransformerWeights::from_tensor_names(&names);
        assert_eq!(
            weights.token_embedding,
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(weights.norm_final, "model.language_model.norm.weight");
        assert_eq!(weights.output, "lm_head.weight");
        assert_eq!(
            weights.layer_weights[0].feed_forward_w2,
            "model.language_model.layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn maps_qwen35_hybrid_names() {
        let spec = ArchitectureSpec {
            model_type: "qwen3_5".to_string(),
            text_layers: vec![TextLayerKind::LinearAttention, TextLayerKind::FullAttention],
            mtp_layers: 1,
            has_vision: true,
            image_token_id: None,
            video_token_id: None,
        };
        let names = vec![
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.1.self_attn.k_proj.weight",
            "model.language_model.layers.1.self_attn.v_proj.weight",
            "model.language_model.layers.1.self_attn.o_proj.weight",
            "model.language_model.layers.1.mlp.gate_proj.weight",
            "model.language_model.layers.1.mlp.down_proj.weight",
            "model.language_model.layers.1.mlp.up_proj.weight",
            "model.language_model.layers.1.input_layernorm.weight",
            "model.language_model.layers.1.post_attention_layernorm.weight",
            "mtp.fc.weight",
        ];
        let weights = HybridTransformerWeights::from_tensor_names(&names, &spec);
        assert_eq!(weights.mtp_fc, Some("mtp.fc.weight".to_string()));
        assert!(weights.has_vision_branch);
        assert!(matches!(
            weights.layer_weights[0].attention,
            HybridAttentionWeights::LinearAttention { .. }
        ));
        assert!(matches!(
            weights.layer_weights[1].attention,
            HybridAttentionWeights::FullAttention { .. }
        ));
    }

    #[test]
    fn falls_back_to_tied_embeddings_for_output() {
        let names = vec![
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
        ];
        let weights = TransformerWeights::from_tensor_names(&names);
        assert_eq!(weights.output, "model.language_model.embed_tokens.weight");
    }
}
