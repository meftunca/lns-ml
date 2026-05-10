use std::{
    collections::{BTreeSet, HashMap},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use safetensors::tensor::{Metadata, TensorInfo};
use safetensors::Dtype;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn shard_paths_from_index(index_path: &Path) -> Result<Vec<PathBuf>> {
    let raw = fs::read_to_string(index_path)
        .with_context(|| format!("cannot read '{}'", index_path.display()))?;
    let index: SafetensorsIndex = serde_json::from_str(&raw)
        .with_context(|| format!("invalid json in '{}'", index_path.display()))?;
    let base_dir = index_path.parent().unwrap_or_else(|| Path::new("."));
    let mut shard_files = BTreeSet::new();
    for relative in index.weight_map.values() {
        shard_files.insert(base_dir.join(relative));
    }
    if shard_files.is_empty() {
        bail!("'{}' contains no shard entries", index_path.display());
    }
    Ok(shard_files.into_iter().collect())
}

fn resolve_input_files(input_path: &Path) -> Result<Vec<PathBuf>> {
    if input_path.is_dir() {
        let index_path = input_path.join("model.safetensors.index.json");
        if index_path.is_file() {
            return shard_paths_from_index(&index_path);
        }

        let mut safetensors_files = Vec::new();
        for entry in fs::read_dir(input_path)
            .with_context(|| format!("cannot read directory '{}'", input_path.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                safetensors_files.push(path);
            }
        }
        safetensors_files.sort();
        if safetensors_files.len() == 1 {
            return Ok(safetensors_files);
        }
        bail!(
            "could not resolve safetensors input from directory '{}': expected model.safetensors.index.json or exactly one .safetensors file",
            input_path.display()
        );
    }

    let file_name = input_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if file_name.ends_with(".safetensors.index.json") {
        return shard_paths_from_index(input_path);
    }
    if input_path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
        return Ok(vec![input_path.to_path_buf()]);
    }

    bail!(
        "unsupported input '{}': expected a safetensors file, safetensors index json, or model directory",
        input_path.display()
    )
}

pub fn visit_input_tensors<S, F>(
    input_path: &Path,
    mut should_visit: S,
    mut visitor: F,
) -> Result<()>
where
    S: FnMut(&str) -> bool,
    F: FnMut(&str, Dtype, &[usize], Vec<u8>) -> Result<()>,
{
    let files = resolve_input_files(input_path)?;
    for file_path in files {
        let mut file = fs::File::open(&file_path)
            .with_context(|| format!("cannot open '{}'", file_path.display()))?;
        let file_len = file
            .metadata()
            .with_context(|| format!("cannot stat '{}'", file_path.display()))?
            .len() as usize;

        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).with_context(|| {
            format!(
                "cannot read safetensors header length from '{}'",
                file_path.display()
            )
        })?;
        let header_len: usize = u64::from_le_bytes(len_bytes)
            .try_into()
            .context("safetensors header length does not fit in usize")?;
        let data_start = 8usize
            .checked_add(header_len)
            .context("safetensors header length overflow")?;
        if data_start > file_len {
            bail!(
                "invalid safetensors header length in '{}'",
                file_path.display()
            );
        }

        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header).with_context(|| {
            format!(
                "cannot read safetensors header from '{}'",
                file_path.display()
            )
        })?;
        let metadata: Metadata = serde_json::from_slice(&header).with_context(|| {
            format!("invalid safetensors metadata in '{}'", file_path.display())
        })?;
        let mut tensors: Vec<(String, TensorInfo)> = metadata
            .tensors()
            .into_iter()
            .map(|(name, info)| (name, info.clone()))
            .collect();
        tensors.sort_by(|(_, left), (_, right)| left.data_offsets.cmp(&right.data_offsets));

        let mut expected_offset = 0usize;
        for (name, info) in &tensors {
            let (start, end) = info.data_offsets;
            if start != expected_offset || end < start {
                bail!("invalid safetensors data offsets for tensor '{name}'");
            }
            let elements = info
                .shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
                .context("safetensors tensor element count overflow")?;
            let expected_len = elements
                .checked_mul(info.dtype.size())
                .context("safetensors tensor byte length overflow")?;
            if end - start != expected_len {
                bail!("invalid safetensors byte length for tensor '{name}'");
            }
            expected_offset = end;
        }
        if data_start + expected_offset != file_len {
            bail!(
                "safetensors metadata does not cover the full file '{}': expected {} bytes, got {}",
                file_path.display(),
                data_start + expected_offset,
                file_len
            );
        }

        for (name, info) in tensors {
            if !should_visit(&name) {
                continue;
            }
            let (start, end) = info.data_offsets;
            let absolute_start = data_start + start;
            let mut data = vec![0u8; end - start];
            file.seek(SeekFrom::Start(absolute_start as u64))
                .with_context(|| {
                    format!(
                        "cannot seek to tensor '{name}' in '{}'",
                        file_path.display()
                    )
                })?;
            file.read_exact(&mut data).with_context(|| {
                format!("cannot read tensor '{name}' from '{}'", file_path.display())
            })?;
            visitor(&name, info.dtype, &info.shape, data)?;
        }
    }
    Ok(())
}

pub fn input_tensor_names(input_path: &Path) -> Result<Vec<String>> {
    let files = resolve_input_files(input_path)?;
    let mut names = Vec::new();
    for file_path in files {
        let mut file = fs::File::open(&file_path)
            .with_context(|| format!("cannot open '{}'", file_path.display()))?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes).with_context(|| {
            format!(
                "cannot read safetensors header length from '{}'",
                file_path.display()
            )
        })?;
        let header_len: usize = u64::from_le_bytes(len_bytes)
            .try_into()
            .context("safetensors header length does not fit in usize")?;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header).with_context(|| {
            format!(
                "cannot read safetensors header from '{}'",
                file_path.display()
            )
        })?;
        let metadata: Metadata = serde_json::from_slice(&header).with_context(|| {
            format!("invalid safetensors metadata in '{}'", file_path.display())
        })?;
        names.extend(metadata.tensors().into_keys());
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::visit_input_tensors;
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

    #[test]
    fn visits_all_sharded_tensors_from_index_file() {
        let dir = temp_dir("sharded-index");
        let shard1 = dir.join("model-00001-of-00002.safetensors");
        let shard2 = dir.join("model-00002-of-00002.safetensors");
        write_shard(&shard1, &[("a.weight", vec![1.0, 2.0])]);
        write_shard(&shard2, &[("b.weight", vec![3.0, 4.0])]);
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

        let mut names = Vec::new();
        visit_input_tensors(
            &dir,
            |_| true,
            |name, _, _, _| {
                names.push(name.to_string());
                Ok(())
            },
        )
        .unwrap();

        names.sort();
        assert_eq!(names, vec!["a.weight".to_string(), "b.weight".to_string()]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn visits_single_safetensors_file_from_directory() {
        let dir = temp_dir("single-shard");
        let shard = dir.join("model.safetensors");
        write_shard(
            &shard,
            &[("tok_embeddings.weight", vec![1.0, 2.0, 3.0, 4.0])],
        );

        let mut names = Vec::new();
        visit_input_tensors(
            &dir,
            |_| true,
            |name, _, _, _| {
                names.push(name.to_string());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(names, vec!["tok_embeddings.weight".to_string()]);

        fs::remove_dir_all(dir).unwrap();
    }
}
