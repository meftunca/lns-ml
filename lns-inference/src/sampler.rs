pub fn sample(logits: &[f32], temperature: f32, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        // Greedy
        let mut max_idx = 0;
        let mut max_val = logits[0];
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return max_idx as u32;
    }

    // Stable softmax trick: subtract max logit
    let mut max_logit = logits[0];
    for &v in logits.iter() {
        if v > max_logit {
            max_logit = v;
        }
    }

    // Apply temperature and softmax
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) / temperature).exp()))
        .collect();

    let sum_exp: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum_exp == 0.0 {
        return 0; // Fallback to unk or something
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum_exp;
    }

    // Sort by probability descending
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-p (Nucleus) sampling
    let mut cumulative_prob = 0.0f32;
    let mut cutoff_idx = probs.len();
    for (i, (_, p)) in probs.iter().enumerate() {
        cumulative_prob += p;
        if cumulative_prob > top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let probs_truncated = &probs[..cutoff_idx];
    let sum_truncated: f32 = probs_truncated.iter().map(|(_, p)| p).sum();

    // Random choice from truncated distribution
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r = rng.gen::<f32>() * sum_truncated;

    let mut acc = 0.0f32;
    for (i, p) in probs_truncated {
        acc += p;
        if r <= acc {
            return *i as u32;
        }
    }

    probs[0].0 as u32 // Fallback
}

/// Min-P sampling: keep tokens where p_i / p_max >= min_p, then sample.
///
/// Compared to top-p, min-p is threshold-relative to the top probability,
/// which naturally scales with the sharpness of the distribution. Typical
/// value: min_p = 0.05..0.10.
pub fn sample_min_p(logits: &[f32], temperature: f32, min_p: f32) -> u32 {
    if temperature <= 0.0 {
        // Greedy
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        return idx as u32;
    }

    // Stable softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_logit) / temperature).exp()))
        .collect();
    let sum_exp: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum_exp == 0.0 {
        return 0;
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum_exp;
    }

    // Min-P filter: keep tokens with prob >= min_p * p_max
    let p_max = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
    let threshold = min_p * p_max;
    let filtered: Vec<(usize, f32)> = probs.into_iter().filter(|(_, p)| *p >= threshold).collect();

    // Fall back to top-1 if filter removes everything
    let candidates = if filtered.is_empty() {
        // Should never happen since p_max >= threshold, but guard anyway
        return 0;
    } else {
        filtered
    };

    let sum_filtered: f32 = candidates.iter().map(|(_, p)| p).sum();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r = rng.gen::<f32>() * sum_filtered;
    let mut acc = 0.0f32;
    for (i, p) in &candidates {
        acc += p;
        if r <= acc {
            return *i as u32;
        }
    }
    candidates[0].0 as u32
}
