use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelJson,
    added_tokens: Option<Vec<AddedToken>>,
}

#[derive(Deserialize)]
struct ModelJson {
    vocab: HashMap<String, u32>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_id: Option<u32>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            token_id: None,
        }
    }

    fn insert(&mut self, token: &str, id: u32) {
        let mut curr = self;
        for c in token.chars() {
            curr = curr.children.entry(c).or_insert(TrieNode::new());
        }
        curr.token_id = Some(id);
    }
}

pub struct Tokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    trie: TrieNode,
    bos_id: u32,
    eos_id: u32,
    has_bos: bool,
    hf_tokenizer: Option<HfTokenizer>,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let content = fs::read_to_string(path_ref)?;
        let json: TokenizerJson = serde_json::from_str(&content)?;
        let hf_tokenizer = HfTokenizer::from_file(path_ref).ok();

        let mut max_id = json.model.vocab.values().cloned().max().unwrap_or(0);
        if let Some(added) = &json.added_tokens {
            let a_max = added.iter().map(|t| t.id).max().unwrap_or(0);
            max_id = max_id.max(a_max);
        }

        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        let mut trie = TrieNode::new();
        let mut token_to_id = json.model.vocab.clone();

        for (token, &id) in &json.model.vocab {
            id_to_token[id as usize] = token.clone();
            trie.insert(token, id);
        }

        if let Some(added) = json.added_tokens {
            for t in added {
                id_to_token[t.id as usize] = t.content.clone();
                token_to_id.insert(t.content.clone(), t.id);
                trie.insert(&t.content, t.id);
            }
        }

        let bos = token_to_id
            .get("<|begin_of_text|>")
            .or(token_to_id.get("<s>"))
            .copied();
        let bos_id = bos.unwrap_or(1);
        let eos_id = *token_to_id
            .get("<|end_of_text|>")
            .or(token_to_id.get("<|endoftext|>"))
            .or(token_to_id.get("</s>"))
            .unwrap_or(&2);

        Ok(Self {
            id_to_token,
            token_to_id,
            trie,
            bos_id,
            eos_id,
            has_bos: bos.is_some(),
            hf_tokenizer,
        })
    }

    pub fn decode(&self, id: u32) -> String {
        if let Some(hf) = &self.hf_tokenizer {
            if let Ok(decoded) = hf.decode(&[id], false) {
                return decoded
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "")
                    .replace("<|end_of_text|>", "")
                    .replace("<|endoftext|>", "")
                    .replace("<|eot_id|>", "");
            }
        }

        if let Some(token) = self.id_to_token.get(id as usize) {
            let s = token
                .replace("\u{2581}", " ")
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .replace("<|end_of_text|>", "")
                .replace("<|endoftext|>", "")
                .replace("<|eot_id|>", "");

            // Byte Fallback handling: detect <0xHH>
            if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
                    return String::from_utf8_lossy(&[byte]).into_owned();
                }
            }
            s
        } else {
            format!("<{}>", id)
        }
    }

    pub fn decode_tokens(&self, ids: &[u32]) -> String {
        if let Some(hf) = &self.hf_tokenizer {
            if let Ok(decoded) = hf.decode(ids, false) {
                return decoded
                    .replace("<|im_start|>", "")
                    .replace("<|im_end|>", "")
                    .replace("<|end_of_text|>", "")
                    .replace("<|endoftext|>", "")
                    .replace("<|eot_id|>", "");
            }
        }

        let mut out = String::new();
        for &id in ids {
            out.push_str(&self.decode(id));
        }
        out
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(hf) = &self.hf_tokenizer {
            if let Ok(encoded) = hf.encode(text, false) {
                return encoded.get_ids().iter().map(|&id| id as u32).collect();
            }
        }

        let mut result = Vec::new();
        if text.is_empty() {
            return result;
        }

        let current = text.replace(' ', "\u{2581}");
        if !current.starts_with('\u{2581}') && self.token_to_id.contains_key("\u{2581}s") {
            // Heuristic for Llama-2 style
            // current = format!("\u{2581}{}", current);
        }

        let chars: Vec<char> = current.chars().collect();
        let mut idx = 0;

        while idx < chars.len() {
            let mut longest_id = None;
            let mut longest_len = 0;
            let mut curr_node = &self.trie;

            for j in idx..chars.len() {
                if let Some(next_node) = curr_node.children.get(&chars[j]) {
                    curr_node = next_node;
                    if let Some(id) = curr_node.token_id {
                        longest_id = Some(id);
                        longest_len = (j - idx) + 1;
                    }
                } else {
                    break;
                }
            }

            if let Some(id) = longest_id {
                result.push(id);
                idx += longest_len;
            } else {
                // Byte-fallback keeps unknown unicode round-trippable instead of dropping it.
                let c = chars[idx];
                let mut pushed = false;
                let mut buf = [0u8; 4];
                for &b in c.encode_utf8(&mut buf).as_bytes() {
                    let key = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.token_to_id.get(&key) {
                        result.push(id);
                        pushed = true;
                    }
                }
                if !pushed {
                    // Last-resort: skip one scalar if no byte-fallback tokens exist.
                }
                idx += 1;
            }
        }
        result
    }

    pub fn bos(&self) -> Option<u32> {
        if self.has_bos {
            Some(self.bos_id)
        } else {
            None
        }
    }
    pub fn eos(&self) -> u32 {
        self.eos_id
    }
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
    pub fn has_token(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;

    #[test]
    fn qwen35_tokenizer_detects_endoftext_eos() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/qwen3.5-4b/tokenizer.json"
        ))
        .unwrap();

        assert_eq!(tokenizer.eos(), 248044);
        assert_eq!(tokenizer.decode(248044), "");
    }
}
