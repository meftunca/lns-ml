use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
use std::{path::PathBuf, process::Command};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn cli_bin() -> &'static str {
    env!("CARGO_BIN_EXE_lns-cli")
}

fn run_cli(args: &[&str]) -> std::process::Output {
    Command::new(cli_bin())
        .current_dir(workspace_root())
        .args(args)
        .output()
        .expect("failed to run lns-cli")
}

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("lns-ml-{name}-{nanos}"));
    fs::create_dir_all(&path).unwrap();
    path
}

fn create_qwen3_fixture() -> PathBuf {
    let dir = temp_dir("qwen3-cli");
    fs::write(
        dir.join("config.json"),
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
    fs::write(
        dir.join("tokenizer.json"),
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
    dir
}

#[test]
fn inspect_includes_compatibility_summary() {
    let output = run_cli(&["inspect", "--model", "models/tinyllama1.1B/tinyllama.lns"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compatibility:"));
    assert!(stdout.contains("Support:  supported"));
}

#[test]
fn doctor_reports_qwen35_directory_as_text_supported() {
    let output = run_cli(&["doctor", "--model", "models/qwen3.5-4b"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Support:    supported"));
    assert!(stdout.contains("multimodal vision"));
    assert!(stdout.contains("MTP auxiliary decoder"));
    assert!(stdout.contains("text runtime ignores"));
}

#[test]
fn doctor_reports_qwen3_directory_as_supported_config() {
    let fixture = create_qwen3_fixture();
    let output = run_cli(&["doctor", "--model", fixture.to_str().unwrap()]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Support:    supported"));
    assert!(stdout.contains("Family:     Qwen3Text"));
    assert!(
        stdout.contains("tensor-name validation was skipped")
            || stdout.contains("Weights:    tensor mapping looks complete")
    );
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn tinyllama_one_shot_chat_smoke() {
    let output = run_cli(&[
        "chat",
        "--model",
        "models/tinyllama1.1B/tinyllama.lns",
        "--prompt",
        "Say hi in one short word.",
        "--temp",
        "0.0",
        "--max-new-tokens",
        "4",
    ]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Assistant:"));
}

fn run_env_chat_smoke(var_name: &str) {
    let model = std::env::var(var_name).expect("required env var missing");
    let output = run_cli(&[
        "chat",
        "--model",
        &model,
        "--prompt",
        "Reply with one word.",
        "--temp",
        "0.0",
        "--max-new-tokens",
        "4",
    ]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Assistant:"));
}

#[test]
#[ignore]
fn llama3_chat_smoke_from_env() {
    run_env_chat_smoke("LNS_LLAMA3_MODEL");
}

#[test]
#[ignore]
fn mistral_chat_smoke_from_env() {
    run_env_chat_smoke("LNS_MISTRAL_MODEL");
}

#[test]
#[ignore]
fn qwen3_chat_smoke_from_env() {
    run_env_chat_smoke("LNS_QWEN3_MODEL");
}
