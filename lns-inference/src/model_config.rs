use anyhow::{bail, Result};
use serde_json::Value;

use crate::transformer::{LinearAttentionConfig, MoeConfig, RopeScalingMode, TransformerConfig};

fn parse_rope_scaling_mode(text_cfg: &Value) -> RopeScalingMode {
    // HuggingFace models use either `rope_scaling` (standard) or `rope_parameters` (Qwen3.5)
    let scaling = text_cfg.get("rope_scaling");
    if let Some(s) = scaling {
        let rope_type = s
            .get("type")
            .or_else(|| s.get("rope_type"))
            .and_then(Value::as_str)
            .unwrap_or("default");
        let factor = s
            .get("factor")
            .and_then(Value::as_f64)
            .map(|x| x as f32)
            .unwrap_or(1.0);
        match rope_type {
            "linear" | "dynamic" => return RopeScalingMode::Linear { factor },
            "llama3" => {
                let original_max_pos = s
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(8192);
                let low_freq_factor = s
                    .get("low_freq_factor")
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(1.0);
                let high_freq_factor = s
                    .get("high_freq_factor")
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(4.0);
                return RopeScalingMode::Llama3 {
                    factor,
                    original_max_pos,
                    low_freq_factor,
                    high_freq_factor,
                };
            }
            "yarn" | "longrope" => {
                let original_max_pos = s
                    .get("original_max_position_embeddings")
                    .and_then(Value::as_u64)
                    .map(|x| x as usize)
                    .unwrap_or(4096);
                let beta_fast = s
                    .get("beta_fast")
                    .or_else(|| s.get("high_freq_factor"))
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(32.0);
                let beta_slow = s
                    .get("beta_slow")
                    .or_else(|| s.get("low_freq_factor"))
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(1.0);
                let attention_factor = s
                    .get("attention_factor")
                    .and_then(Value::as_f64)
                    .map(|x| x as f32)
                    .unwrap_or(1.0);
                return RopeScalingMode::Yarn {
                    factor,
                    original_max_pos,
                    beta_fast,
                    beta_slow,
                    attention_factor,
                };
            }
            _ => {}
        }
    }
    RopeScalingMode::Default
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Mistral,
    Qwen3Text,
    Qwen35Text,
    /// Mixtral-style sparse MoE (block_sparse_moe experts).
    MixtralMoe,
    /// Qwen2/Qwen3-style sparse MoE (mlp.experts + optional shared expert).
    Qwen2Moe,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedModelConfig {
    pub family: ModelFamily,
    pub transformer: TransformerConfig,
    pub architecture: ArchitectureSpec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextLayerKind {
    FullAttention,
    LinearAttention,
    MlpOnly,
    Unknown(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureSpec {
    pub model_type: String,
    pub text_layers: Vec<TextLayerKind>,
    pub mtp_layers: usize,
    pub has_vision: bool,
    /// Token ID used as a placeholder in the token stream for image patches.
    /// When `Some`, the engine will reject any `forward()` call with this token.
    pub image_token_id: Option<u32>,
    /// Token ID used as a placeholder in the token stream for video patches.
    /// When `Some`, the engine will reject any `forward()` call with this token.
    pub video_token_id: Option<u32>,
}

fn as_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|x| x as usize)
}

fn as_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key)?.as_f64().map(|x| x as f32)
}

fn unsupported_features(root: &Value, text_cfg: &Value, _model_type: &str) -> Vec<&'static str> {
    let has_linear_attention = text_cfg
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter().any(|x| {
                x.as_str()
                    .map(|s| s.contains("linear_attention"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    let has_vision = root.get("vision_config").is_some() || root.get("image_token_id").is_some();
    let has_moe = text_cfg.get("num_local_experts").is_some()
        || text_cfg.get("moe_intermediate_size").is_some()
        || text_cfg.get("num_experts").is_some();
    let _ = has_moe; // MoE is now supported — keep detection for unsupported_feature_details
    let has_mtp = text_cfg
        .get("mtp_num_hidden_layers")
        .and_then(Value::as_u64)
        .map(|n| n > 0)
        .unwrap_or(false);

    let mut features = Vec::new();
    if has_linear_attention {
        features.push("hybrid linear-attention");
    }
    if has_vision {
        features.push("multimodal vision");
    }
    // MoE routing is now supported — removed from unsupported list
    if has_mtp {
        features.push("MTP auxiliary decoder");
    }
    features
}

pub fn inspect_architecture(root: &Value) -> ArchitectureSpec {
    let model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let text_cfg = root.get("text_config").unwrap_or(root);
    let n_layers = as_usize(text_cfg, "num_hidden_layers").unwrap_or(0);
    let text_layers = text_cfg
        .get("layer_types")
        .and_then(Value::as_array)
        .map(|layers| {
            layers
                .iter()
                .map(|layer| match layer.as_str().unwrap_or("unknown") {
                    "full_attention" => TextLayerKind::FullAttention,
                    "linear_attention" => TextLayerKind::LinearAttention,
                    "mlp_only" => TextLayerKind::MlpOnly,
                    other => TextLayerKind::Unknown(other.to_string()),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![TextLayerKind::FullAttention; n_layers]);

    ArchitectureSpec {
        model_type,
        text_layers,
        mtp_layers: text_cfg
            .get("mtp_num_hidden_layers")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(0),
        has_vision: root.get("vision_config").is_some() || root.get("image_token_id").is_some(),
        image_token_id: root
            .get("image_token_id")
            .and_then(Value::as_u64)
            .map(|x| x as u32),
        video_token_id: root
            .get("video_token_id")
            .and_then(Value::as_u64)
            .map(|x| x as u32),
    }
}

pub fn unsupported_feature_details(root: &Value) -> Vec<String> {
    let model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let text_cfg = root.get("text_config").unwrap_or(root);
    let mut details = Vec::new();

    if unsupported_features(root, text_cfg, model_type).contains(&"hybrid linear-attention") {
        if let Some(layer_types) = text_cfg.get("layer_types").and_then(Value::as_array) {
            let total = layer_types.len();
            let linear = layer_types
                .iter()
                .filter(|kind| {
                    kind.as_str()
                        .map(|s| s.contains("linear_attention"))
                        .unwrap_or(false)
                })
                .count();
            let full = layer_types
                .iter()
                .filter(|kind| {
                    kind.as_str()
                        .map(|s| s.contains("full_attention"))
                        .unwrap_or(false)
                })
                .count();
            let interval = text_cfg
                .get("full_attention_interval")
                .and_then(Value::as_u64)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            details.push(format!(
                "attention schedule: {linear}/{total} linear-attention layers, {full}/{total} full-attention layers, full interval={interval}"
            ));
        }
    }

    if unsupported_features(root, text_cfg, model_type).contains(&"multimodal vision") {
        if let Some(vision) = root.get("vision_config") {
            let depth = vision.get("depth").and_then(Value::as_u64).unwrap_or(0);
            let hidden = vision
                .get("hidden_size")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let image_token_id = root
                .get("image_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let video_token_id = root
                .get("video_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            details.push(format!(
                "vision branch: depth={depth}, hidden_size={hidden}, image_token_id={image_token_id}, video_token_id={video_token_id}"
            ));
        }
    }

    if unsupported_features(root, text_cfg, model_type).contains(&"MoE routing") {
        let experts = text_cfg
            .get("num_local_experts")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let moe_hidden = text_cfg
            .get("moe_intermediate_size")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        details.push(format!(
            "moe routing: local_experts={experts}, moe_intermediate_size={moe_hidden}"
        ));
    }

    if unsupported_features(root, text_cfg, model_type).contains(&"MTP auxiliary decoder") {
        let mtp_layers = text_cfg
            .get("mtp_num_hidden_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let dedicated_embeddings = text_cfg
            .get("mtp_use_dedicated_embeddings")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        details.push(format!(
            "mtp path: hidden_layers={mtp_layers}, dedicated_embeddings={dedicated_embeddings}"
        ));
    }

    details
}

pub fn parse_model_config(root: &Value) -> Result<NormalizedModelConfig> {
    let model_type = root
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let text_cfg = root.get("text_config").unwrap_or(root);

    let unsupported = unsupported_features(root, text_cfg, model_type);
    if !unsupported.is_empty() && model_type != "qwen3_5" {
        bail!(
            "Unsupported architecture for current chat engine: these features are not supported yet: {}.",
            unsupported.join(", ")
        );
    }

    let family = match model_type {
        "llama" => {
            let sliding_window = text_cfg.get("sliding_window").and_then(Value::as_u64);
            if sliding_window.is_some() {
                ModelFamily::Mistral
            } else {
                ModelFamily::Llama
            }
        }
        "mistral" => ModelFamily::Mistral,
        "mixtral" => ModelFamily::MixtralMoe,
        "qwen2_moe" | "qwen3_moe" | "qwen2_moe_vl" => ModelFamily::Qwen2Moe,
        "qwen3" => ModelFamily::Qwen3Text,
        "qwen3_5" => ModelFamily::Qwen35Text,
        other => bail!("Unsupported model_type for current engine: {other}"),
    };

    let dim =
        as_usize(text_cfg, "hidden_size").ok_or_else(|| anyhow::anyhow!("missing hidden_size"))?;
    let hidden_dim = as_usize(text_cfg, "intermediate_size")
        .ok_or_else(|| anyhow::anyhow!("missing intermediate_size"))?;
    let n_layers = as_usize(text_cfg, "num_hidden_layers")
        .ok_or_else(|| anyhow::anyhow!("missing num_hidden_layers"))?;
    let n_heads = as_usize(text_cfg, "num_attention_heads")
        .ok_or_else(|| anyhow::anyhow!("missing num_attention_heads"))?;
    let n_kv_heads = as_usize(text_cfg, "num_key_value_heads").unwrap_or(n_heads);
    let vocab_size =
        as_usize(text_cfg, "vocab_size").ok_or_else(|| anyhow::anyhow!("missing vocab_size"))?;
    let eps = as_f32(text_cfg, "rms_norm_eps")
        .or_else(|| as_f32(text_cfg, "layer_norm_epsilon"))
        .unwrap_or(1e-5);
    let rope_theta = as_f32(text_cfg, "rope_theta")
        .or_else(|| {
            text_cfg
                .get("rope_parameters")
                .and_then(|r| r.get("rope_theta"))
                .and_then(Value::as_f64)
                .map(|x| x as f32)
        })
        .or_else(|| as_f32(text_cfg, "rope_freq_base"))
        .unwrap_or(10000.0);
    let rope_scaling_factor = text_cfg
        .get("rope_scaling")
        .and_then(|r| r.get("factor"))
        .and_then(Value::as_f64)
        .map(|x| x as f32)
        .unwrap_or(1.0);
    let attention_head_dim = as_usize(text_cfg, "head_dim").unwrap_or(dim / n_heads);
    let qk_norm = text_cfg.get("q_norm").is_some()
        || text_cfg
            .get("use_qk_norm")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        || model_type == "qwen3"
        || model_type == "qwen3_5";
    let zero_centered_rmsnorm = model_type == "qwen3_5";
    let gated_query_attention = model_type == "qwen3_5";
    let tie_word_embeddings = text_cfg
        .get("tie_word_embeddings")
        .or_else(|| root.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let partial_rotary_factor = text_cfg
        .get("rope_parameters")
        .and_then(|r| r.get("partial_rotary_factor"))
        .and_then(Value::as_f64)
        .map(|x| x as f32)
        .unwrap_or(1.0);

    let linear_attention = match model_type {
        "qwen3_5" => Some(LinearAttentionConfig {
            num_key_heads: as_usize(text_cfg, "linear_num_key_heads")
                .ok_or_else(|| anyhow::anyhow!("missing linear_num_key_heads"))?,
            num_value_heads: as_usize(text_cfg, "linear_num_value_heads")
                .ok_or_else(|| anyhow::anyhow!("missing linear_num_value_heads"))?,
            key_head_dim: as_usize(text_cfg, "linear_key_head_dim")
                .ok_or_else(|| anyhow::anyhow!("missing linear_key_head_dim"))?,
            value_head_dim: as_usize(text_cfg, "linear_value_head_dim")
                .ok_or_else(|| anyhow::anyhow!("missing linear_value_head_dim"))?,
            conv_kernel_size: as_usize(text_cfg, "linear_conv_kernel_dim")
                .ok_or_else(|| anyhow::anyhow!("missing linear_conv_kernel_dim"))?,
        }),
        _ => None,
    };

    let rope_mode = parse_rope_scaling_mode(text_cfg);
    let sliding_window = text_cfg
        .get("sliding_window")
        .and_then(Value::as_u64)
        .map(|x| x as usize);
    let architecture = inspect_architecture(root);

    // MoE config — parsed for Mixtral and Qwen2/3-MoE families.
    let moe = if let Some(n_experts) =
        as_usize(text_cfg, "num_local_experts").or_else(|| as_usize(text_cfg, "num_experts"))
    {
        let n_experts_per_tok = as_usize(text_cfg, "num_experts_per_tok")
            .or_else(|| as_usize(text_cfg, "num_selected_experts"))
            .unwrap_or(2);
        let moe_hidden_dim = as_usize(text_cfg, "moe_intermediate_size").unwrap_or(hidden_dim);
        let n_shared_experts = as_usize(text_cfg, "num_shared_experts")
            .or_else(|| {
                if text_cfg.get("shared_expert_intermediate_size").is_some() {
                    Some(1)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        Some(MoeConfig {
            n_experts,
            n_experts_per_tok,
            moe_hidden_dim,
            n_shared_experts,
        })
    } else {
        None
    };

    Ok(NormalizedModelConfig {
        family,
        transformer: TransformerConfig {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            eps,
            rope_theta,
            rope_scaling_factor,
            rope_mode,
            attention_head_dim,
            partial_rotary_factor,
            qk_norm,
            zero_centered_rmsnorm,
            gated_query_attention,
            tie_word_embeddings,
            linear_attention,
            sliding_window,
            moe,
        },
        architecture,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        inspect_architecture, parse_model_config, unsupported_feature_details,
        unsupported_features, ModelFamily, TextLayerKind,
    };
    use serde_json::json;

    #[test]
    fn parses_llama3_config() {
        let root = json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        });
        let parsed = parse_model_config(&root).unwrap();
        assert_eq!(parsed.family, ModelFamily::Llama);
        assert_eq!(parsed.transformer.dim, 4096);
        assert_eq!(parsed.transformer.n_kv_heads, 8);
        assert_eq!(parsed.transformer.rope_theta, 500000.0);
        assert_eq!(parsed.architecture.text_layers.len(), 32);
    }

    #[test]
    fn parses_mistral_config() {
        let root = json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window": 4096
        });
        let parsed = parse_model_config(&root).unwrap();
        assert_eq!(parsed.family, ModelFamily::Mistral);
        assert_eq!(parsed.transformer.hidden_dim, 14336);
        assert_eq!(parsed.transformer.sliding_window, Some(4096));
        assert_eq!(parsed.architecture.text_layers.len(), 32);
    }

    #[test]
    fn parses_qwen3_text_config() {
        let root = json!({
            "model_type": "qwen3",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0
        });
        let parsed = parse_model_config(&root).unwrap();
        assert_eq!(parsed.family, ModelFamily::Qwen3Text);
        assert_eq!(parsed.transformer.vocab_size, 151936);
        assert_eq!(parsed.transformer.rope_theta, 1_000_000.0);
        assert_eq!(parsed.transformer.n_layers, 36);
        assert_eq!(parsed.architecture.text_layers.len(), 36);
    }

    #[test]
    fn parses_qwen35_runtime_config_with_multimodal_metadata() {
        let root = json!({
            "model_type": "qwen3_5",
            "vision_config": { "hidden_size": 1024 },
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "linear_num_key_heads": 8,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention", "full_attention"],
                "mtp_num_hidden_layers": 1
            }
        });
        let parsed = parse_model_config(&root).unwrap();
        assert_eq!(parsed.family, ModelFamily::Qwen35Text);
        assert!(parsed.architecture.has_vision);
        assert_eq!(parsed.architecture.mtp_layers, 1);
        assert!(parsed.transformer.linear_attention.is_some());
    }

    #[test]
    fn parses_qwen35_nested_rope_parameters() {
        let root = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25
                }
            }
        });
        let parsed = parse_model_config(&root).unwrap();
        assert_eq!(parsed.transformer.rope_theta, 10_000_000.0);
        assert!((parsed.transformer.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert_eq!(parsed.transformer.attention_head_dim, 256);
        assert!(parsed.transformer.zero_centered_rmsnorm);
        assert!(parsed.transformer.gated_query_attention);
    }

    #[test]
    fn rejects_llama4_multimodal_config() {
        // MoE with vision — MoE is now supported, but multimodal vision is still rejected.
        let root = json!({
            "model_type": "llama",
            "vision_config": { "hidden_size": 1024 },
            "hidden_size": 5120,
            "intermediate_size": 8192,
            "num_hidden_layers": 48,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "num_local_experts": 16,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        });
        let err = parse_model_config(&root).unwrap_err().to_string();
        assert!(
            !err.contains("MoE routing"),
            "MoE routing should now be supported"
        );
        assert!(err.contains("multimodal vision"));
    }

    #[test]
    fn extracts_precise_unsupported_feature_list() {
        let root = json!({
            "model_type": "qwen3_5",
            "vision_config": { "hidden_size": 1024 },
            "text_config": {
                "layer_types": ["linear_attention", "full_attention"],
                "mtp_num_hidden_layers": 1
            }
        });
        let features = unsupported_features(&root, root.get("text_config").unwrap(), "qwen3_5");
        assert_eq!(
            features,
            vec![
                "hybrid linear-attention",
                "multimodal vision",
                "MTP auxiliary decoder",
            ]
        );
    }

    #[test]
    fn extracts_unsupported_feature_details() {
        let root = json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024
            },
            "text_config": {
                "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
                "full_attention_interval": 2,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }
        });
        let details = unsupported_feature_details(&root);
        assert!(details
            .iter()
            .any(|line| line.contains("2/4 linear-attention layers")));
        assert!(details
            .iter()
            .any(|line| line.contains("vision branch: depth=24")));
        assert!(details
            .iter()
            .any(|line| line.contains("mtp path: hidden_layers=1")));
    }

    #[test]
    fn inspects_qwen35_hybrid_architecture() {
        let root = json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "full_attention", "mlp_only"],
                "mtp_num_hidden_layers": 1
            }
        });
        let spec = inspect_architecture(&root);
        assert_eq!(spec.model_type, "qwen3_5");
        assert_eq!(spec.text_layers.len(), 4);
        assert_eq!(spec.text_layers[0], TextLayerKind::LinearAttention);
        assert_eq!(spec.text_layers[2], TextLayerKind::FullAttention);
        assert_eq!(spec.text_layers[3], TextLayerKind::MlpOnly);
        assert_eq!(spec.mtp_layers, 1);
        assert!(spec.has_vision);
        assert_eq!(spec.image_token_id, Some(248056));
        assert_eq!(spec.video_token_id, None);
    }

    #[test]
    fn inspects_vision_token_ids() {
        let root = json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_config": { "hidden_size": 1024 },
            "text_config": { "num_hidden_layers": 2 }
        });
        let spec = inspect_architecture(&root);
        assert!(spec.has_vision);
        assert_eq!(spec.image_token_id, Some(248056));
        assert_eq!(spec.video_token_id, Some(248057));
    }
}
