use serde_json::Value;

use crate::{
    model_config::{
        inspect_architecture, parse_model_config, unsupported_feature_details, ModelFamily,
        NormalizedModelConfig,
    },
    transformer::{HybridAttentionWeights, HybridTransformerWeights, TransformerWeights},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportTier {
    Supported,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateRecommendation {
    Llama2,
    Llama3,
    ChatML,
    Plain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompatibilityReport {
    pub tier: SupportTier,
    pub family: Option<ModelFamily>,
    pub config: Option<NormalizedModelConfig>,
    pub template: TemplateRecommendation,
    pub issues: Vec<String>,
    pub details: Vec<String>,
    pub missing_tensors: Vec<String>,
}

fn has_name(names: &[&str], needle: &str) -> bool {
    names.iter().any(|name| *name == needle)
}

pub fn recommend_template(tokens: &[&str]) -> TemplateRecommendation {
    if has_name(tokens, "<|im_start|>") && has_name(tokens, "<|im_end|>") {
        TemplateRecommendation::ChatML
    } else if has_name(tokens, "<|start_header_id|>") && has_name(tokens, "<|eot_id|>") {
        TemplateRecommendation::Llama3
    } else if has_name(tokens, "[INST]") && has_name(tokens, "[/INST]") {
        TemplateRecommendation::Llama2
    } else {
        TemplateRecommendation::Plain
    }
}

fn collect_missing_tensors(tensor_names: &[&str], config: &NormalizedModelConfig) -> Vec<String> {
    let weights = TransformerWeights::from_tensor_names(tensor_names);
    let mut missing = Vec::new();

    for required in [
        weights.token_embedding.as_str(),
        weights.norm_final.as_str(),
        weights.output.as_str(),
    ] {
        if !has_name(tensor_names, required) {
            missing.push(required.to_string());
        }
    }

    for layer_idx in 0..config.transformer.n_layers {
        let Some(layer) = weights.layer_weights.get(layer_idx) else {
            missing.push(format!("layer {layer_idx} mapping could not be derived"));
            continue;
        };

        for required in [
            layer.attention_wq.as_str(),
            layer.attention_wk.as_str(),
            layer.attention_wv.as_str(),
            layer.attention_wo.as_str(),
            layer.feed_forward_w1.as_str(),
            layer.feed_forward_w2.as_str(),
            layer.feed_forward_w3.as_str(),
            layer.attention_norm.as_str(),
            layer.ffn_norm.as_str(),
        ] {
            if !has_name(tensor_names, required) {
                missing.push(required.to_string());
            }
        }
    }

    missing
}

fn collect_missing_hybrid_tensors(root: &Value, tensor_names: &[&str]) -> Vec<String> {
    if tensor_names.is_empty() {
        return Vec::new();
    }

    let spec = inspect_architecture(root);
    let weights = HybridTransformerWeights::from_tensor_names(tensor_names, &spec);
    let mut missing = Vec::new();

    for required in [
        weights.token_embedding.as_str(),
        weights.norm_final.as_str(),
        weights.output.as_str(),
    ] {
        if !has_name(tensor_names, required) {
            missing.push(required.to_string());
        }
    }

    for layer in &weights.layer_weights {
        match &layer.attention {
            HybridAttentionWeights::FullAttention {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
            } => {
                for required in [
                    q_proj.as_str(),
                    k_proj.as_str(),
                    v_proj.as_str(),
                    o_proj.as_str(),
                ] {
                    if !has_name(tensor_names, required) {
                        missing.push(required.to_string());
                    }
                }
                for required in [q_norm.as_deref(), k_norm.as_deref()].into_iter().flatten() {
                    if !has_name(tensor_names, required) {
                        missing.push(required.to_string());
                    }
                }
            }
            HybridAttentionWeights::LinearAttention {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
                out_proj,
                conv1d_weight,
                conv1d_bias,
                norm,
                a_log,
                dt_bias,
            } => {
                for required in [
                    in_proj_qkv.as_str(),
                    in_proj_z.as_str(),
                    in_proj_b.as_str(),
                    in_proj_a.as_str(),
                    out_proj.as_str(),
                    conv1d_weight.as_str(),
                    norm.as_str(),
                    a_log.as_str(),
                    dt_bias.as_str(),
                ] {
                    if !has_name(tensor_names, required) {
                        missing.push(required.to_string());
                    }
                }
                if let Some(conv1d_bias) = conv1d_bias {
                    if !has_name(tensor_names, conv1d_bias) {
                        missing.push(conv1d_bias.to_string());
                    }
                }
            }
        }

        for required in [
            layer.feed_forward_w1.as_str(),
            layer.feed_forward_w2.as_str(),
            layer.feed_forward_w3.as_str(),
            layer.attention_norm.as_str(),
            layer.ffn_norm.as_str(),
        ] {
            if !has_name(tensor_names, required) {
                missing.push(required.to_string());
            }
        }
    }

    missing
}

fn hybrid_mapping_details(root: &Value, tensor_names: &[&str]) -> Vec<String> {
    if tensor_names.is_empty() {
        return Vec::new();
    }

    let spec = inspect_architecture(root);
    let linear_layers = spec
        .text_layers
        .iter()
        .filter(|kind| matches!(kind, crate::model_config::TextLayerKind::LinearAttention))
        .count();
    let full_layers = spec
        .text_layers
        .iter()
        .filter(|kind| matches!(kind, crate::model_config::TextLayerKind::FullAttention))
        .count();
    let mlp_only_layers = spec
        .text_layers
        .iter()
        .filter(|kind| matches!(kind, crate::model_config::TextLayerKind::MlpOnly))
        .count();
    let missing = collect_missing_hybrid_tensors(root, tensor_names);

    let mut details = vec![format!(
        "hybrid text mapping: layers={} (linear={}, full={}, mlp-only={}), missing_named_tensors={}",
        spec.text_layers.len(),
        linear_layers,
        full_layers,
        mlp_only_layers,
        missing.len()
    )];
    if spec.mtp_layers > 0 {
        details.push(format!(
            "mtp mapping: layers={}, fc_present={}",
            spec.mtp_layers,
            has_name(tensor_names, "mtp.fc.weight")
        ));
    }
    details
}

pub fn inspect_compatibility(
    root: &Value,
    tensor_names: &[&str],
    tokenizer_tokens: &[&str],
) -> CompatibilityReport {
    match parse_model_config(root) {
        Ok(config) => {
            let mut issues = Vec::new();
            let hybrid_text = config.transformer.linear_attention.is_some()
                || config.architecture.text_layers.iter().any(|kind| {
                    matches!(kind, crate::model_config::TextLayerKind::LinearAttention)
                });
            let missing_tensors = if tensor_names.is_empty() {
                issues.push(
                    "No .lns archive provided; tensor-name validation was skipped.".to_string(),
                );
                Vec::new()
            } else if hybrid_text {
                collect_missing_hybrid_tensors(root, tensor_names)
            } else {
                collect_missing_tensors(tensor_names, &config)
            };

            if config.architecture.has_vision {
                let img_id = config
                    .architecture
                    .image_token_id
                    .map(|id| format!("image_token_id={id}"))
                    .unwrap_or_default();
                let vid_id = config
                    .architecture
                    .video_token_id
                    .map(|id| format!(", video_token_id={id}"))
                    .unwrap_or_default();
                let ids = if img_id.is_empty() {
                    String::new()
                } else {
                    format!(" ({img_id}{vid_id})")
                };
                issues.push(format!(
                    "This model includes a multimodal vision branch{ids}; \
                     feeding vision placeholder tokens will return an error at runtime."
                ));
            }
            if config.architecture.mtp_layers > 0 {
                issues.push(format!(
                    "This model includes an MTP auxiliary decoder ({} layer{}); the current text runtime ignores it.",
                    config.architecture.mtp_layers,
                    if config.architecture.mtp_layers == 1 { "" } else { "s" }
                ));
            }

            if !missing_tensors.is_empty() {
                issues.push(format!(
                    "Expected {} transformer layers but some required tensors are missing from the archive.",
                    config.transformer.n_layers
                ));
            }

            CompatibilityReport {
                tier: if missing_tensors.is_empty() {
                    SupportTier::Supported
                } else {
                    SupportTier::Unsupported
                },
                family: Some(config.family),
                config: Some(config),
                template: recommend_template(tokenizer_tokens),
                issues,
                details: if hybrid_text {
                    hybrid_mapping_details(root, tensor_names)
                } else {
                    Vec::new()
                },
                missing_tensors,
            }
        }
        Err(err) => CompatibilityReport {
            tier: SupportTier::Unsupported,
            family: None,
            config: None,
            template: recommend_template(tokenizer_tokens),
            issues: vec![err.to_string()],
            details: {
                let mut details = unsupported_feature_details(root);
                details.extend(hybrid_mapping_details(root, tensor_names));
                details
            },
            missing_tensors: collect_missing_hybrid_tensors(root, tensor_names),
        },
    }
}
#[cfg(test)]
mod tests {
    use super::{inspect_compatibility, SupportTier, TemplateRecommendation};
    use serde_json::json;

    #[test]
    fn marks_llama3_archive_supported() {
        let root = json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0
        });
        let tensors = vec![
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
        let tokens = vec!["<|start_header_id|>", "<|eot_id|>"];
        let report = inspect_compatibility(&root, &tensors, &tokens);
        assert_eq!(report.tier, SupportTier::Supported);
        assert_eq!(report.template, TemplateRecommendation::Llama3);
    }

    #[test]
    fn marks_missing_tensors_unsupported() {
        let root = json!({
            "model_type": "qwen3",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0
        });
        let tensors = vec!["model.language_model.embed_tokens.weight"];
        let tokens = vec!["<|im_start|>", "<|im_end|>"];
        let report = inspect_compatibility(&root, &tensors, &tokens);
        assert_eq!(report.tier, SupportTier::Unsupported);
        assert_eq!(report.template, TemplateRecommendation::ChatML);
        assert!(!report.missing_tensors.is_empty());
    }

    #[test]
    fn reports_detailed_unsupported_diagnostics() {
        let root = json!({
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "vision_config": { "depth": 24, "hidden_size": 1024 },
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "linear_num_key_heads": 8,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 248320,
                "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention", "full_attention"],
                "full_attention_interval": 4,
                "mtp_num_hidden_layers": 1
            }
        });
        let report = inspect_compatibility(&root, &[], &["<|im_start|>", "<|im_end|>"]);
        assert_eq!(report.tier, SupportTier::Supported);
        assert!(report
            .issues
            .iter()
            .any(|line| line.contains("tensor-name validation was skipped")));
        assert!(report
            .issues
            .iter()
            .any(|line| line.contains("text runtime ignores")));
    }

    #[test]
    fn validates_hybrid_tensor_names_when_available() {
        let root = json!({
            "model_type": "qwen3_5",
            "vision_config": { "hidden_size": 1024 },
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
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
        let tensors = vec![
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.post_attention_layernorm.weight",
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.1.self_attn.k_proj.weight",
            "model.language_model.layers.1.self_attn.v_proj.weight",
            "model.language_model.layers.1.self_attn.o_proj.weight",
            "model.language_model.layers.1.self_attn.q_norm.weight",
            "model.language_model.layers.1.self_attn.k_norm.weight",
            "model.language_model.layers.1.mlp.gate_proj.weight",
            "model.language_model.layers.1.mlp.down_proj.weight",
            "model.language_model.layers.1.mlp.up_proj.weight",
            "model.language_model.layers.1.input_layernorm.weight",
            "model.language_model.layers.1.post_attention_layernorm.weight",
            "mtp.fc.weight",
        ];
        let report = inspect_compatibility(&root, &tensors, &["<|im_start|>", "<|im_end|>"]);
        assert_eq!(report.tier, SupportTier::Supported);
        assert!(report.missing_tensors.is_empty());
        assert!(report
            .issues
            .iter()
            .any(|line| line.contains("multimodal vision")));
        assert!(report
            .details
            .iter()
            .any(|line| line.contains("hybrid text mapping:")));
    }
}
