use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use bytemuck::cast_slice;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lns-ml-{name}-{nanos}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn write_tokenizer_json(path: &Path) {
    fs::write(
        path,
        r#"{
            "model": {
                "vocab": {
                    "<|im_start|>": 0,
                    "<|im_end|>": 1,
                    "<|end_of_text|>": 2,
                    "hello": 3
                }
            },
            "added_tokens": []
        }"#,
    )
    .unwrap();
}

fn write_config_json(path: &Path) {
    fs::write(
        path,
        r#"{
            "model_type": "qwen3",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 16,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000.0
        }"#,
    )
    .unwrap();
}

fn write_shard(path: &Path, tensors: &[(&str, Vec<f32>)]) {
    let data: Vec<(String, Vec<f32>)> = tensors
        .iter()
        .map(|(name, values)| ((*name).to_string(), values.clone()))
        .collect();
    let mut views = HashMap::new();
    for (name, values) in &data {
        let view = TensorView::new(Dtype::F32, vec![values.len()], cast_slice(values)).unwrap();
        views.insert(name.clone(), view);
    }
    serialize_to_file(&views, &None, path).unwrap();
}

fn create_qwen3_fixture() -> PathBuf {
    let dir = temp_dir("qwen3-convert");
    write_config_json(&dir.join("config.json"));
    write_tokenizer_json(&dir.join("tokenizer.json"));

    let shard1 = dir.join("model-00001-of-00002.safetensors");
    let shard2 = dir.join("model-00002-of-00002.safetensors");
    write_shard(
        &shard1,
        &[
            ("model.language_model.embed_tokens.weight", vec![0.1; 32]),
            ("model.language_model.norm.weight", vec![1.0; 32]),
            ("lm_head.weight", vec![0.2; 32]),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                vec![0.3; 32],
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                vec![0.4; 32],
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight",
                vec![0.5; 32],
            ),
        ],
    );
    write_shard(
        &shard2,
        &[
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                vec![0.6; 32],
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                vec![0.7; 32],
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight",
                vec![0.8; 32],
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                vec![0.9; 32],
            ),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                vec![1.0; 32],
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                vec![1.1; 32],
            ),
        ],
    );

    fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
            "weight_map": {
                "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.language_model.norm.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.self_attn.o_proj.weight": "model-00002-of-00002.safetensors",
                "model.language_model.layers.0.mlp.gate_proj.weight": "model-00002-of-00002.safetensors",
                "model.language_model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
                "model.language_model.layers.0.mlp.up_proj.weight": "model-00002-of-00002.safetensors",
                "model.language_model.layers.0.input_layernorm.weight": "model-00002-of-00002.safetensors",
                "model.language_model.layers.0.post_attention_layernorm.weight": "model-00002-of-00002.safetensors"
            }
        }"#,
    )
    .unwrap();

    dir
}

#[test]
fn converts_sharded_qwen3_directory_input() {
    let fixture = create_qwen3_fixture();
    let output_path = fixture.join("model.lns");

    let status = Command::new(env!("CARGO_BIN_EXE_lns-convert"))
        .arg("--input")
        .arg(&fixture)
        .arg("--output")
        .arg(&output_path)
        .arg("--quant")
        .arg("F16")
        .status()
        .expect("failed to run lns-convert");
    assert!(status.success());

    let bytes = fs::read(&output_path).unwrap();
    let archived = lns_core::format::check_archived_model(&bytes).unwrap();
    assert_eq!(archived.tensors.len(), 12);
    assert!(archived
        .tensors
        .iter()
        .any(|t| t.name.as_str() == "model.language_model.embed_tokens.weight"));
    assert!(archived
        .tensors
        .iter()
        .any(|t| t.name.as_str() == "model.language_model.layers.0.mlp.up_proj.weight"));

    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn converts_q4hq_payload_with_v4_archive() {
    let fixture = temp_dir("q4hq-convert");
    let shard = fixture.join("model.safetensors");
    let output_path = fixture.join("model.lns");
    let values: Vec<f32> = (0..256)
        .map(|index| (index as f32 * 0.037).sin() * 0.75)
        .collect();
    write_shard(
        &shard,
        &[(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            values.clone(),
        )],
    );

    let status = Command::new(env!("CARGO_BIN_EXE_lns-convert"))
        .arg("--input")
        .arg(&shard)
        .arg("--output")
        .arg(&output_path)
        .arg("--quant")
        .arg("Q4_HQ")
        .status()
        .expect("failed to run lns-convert");
    assert!(status.success());

    let bytes = fs::read(&output_path).unwrap();
    let archived = lns_core::format::check_archived_model(&bytes).unwrap();
    assert_eq!(archived.version, lns_core::HQ_FORMAT_VERSION);
    assert_eq!(archived.tensors.len(), 1);

    let tensor = &archived.tensors[0];
    assert_eq!(tensor.quant_type, lns_core::QuantType::Q4HQ.as_u8());
    let header =
        lns_core::parse_hq_payload_header(tensor.data.as_slice(), lns_core::QuantType::Q4HQ)
            .unwrap();
    assert_eq!(header.block_bytes, lns_core::Q4HQ_BLOCK_BYTES);
    assert_eq!(header.block_data_off, lns_core::HQ_CANONICAL_BLOCK_DATA_OFF);

    let decoded = tensor.to_f32().unwrap();
    assert_eq!(decoded.len(), values.len());

    fs::remove_dir_all(fixture).unwrap();
}
