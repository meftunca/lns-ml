use lns_metal::*;

fn cpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dim = x.len();
    let mut ss = 0.0f32;
    for &val in x {
        ss += val * val;
    }
    let inv_rms = 1.0 / (ss / dim as f32 + eps).sqrt();
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        out[i] = x[i] * inv_rms * w[i];
    }
    out
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_rmsnorm_consistency() {
    let dim = 1024;
    let x: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let w: Vec<f32> = vec![1.0f32; dim];
    let eps = 1e-5;

    let cpu_res = cpu_rmsnorm(&x, &w, eps);
    let ctx = get_metal_context().expect("Metal context");
    let gpu_res = ctx.dispatch_rmsnorm(&x, &w, eps).expect("GPU dispatch");

    for i in 0..dim {
        let diff = (cpu_res[i] - gpu_res[i]).abs();
        assert!(diff < 1e-5);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_rope_consistency() {
    let dim = 128;
    let head_dim = 64;
    let pos = 10;
    let mut q: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.2).cos()).collect();
    let mut k: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).sin()).collect();

    let mut q_cpu = q.clone();
    let mut k_cpu = k.clone();

    // CPU RoPE
    for head in 0..(dim / head_dim) {
        for i in 0..(head_dim / 2) {
            let theta = pos as f32 * (10000.0f32).powf(-(2.0 * i as f32) / head_dim as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            let q_idx = head * head_dim + i;
            let q0 = q_cpu[q_idx];
            let q1 = q_cpu[q_idx + head_dim / 2];
            q_cpu[q_idx] = q0 * cos_t - q1 * sin_t;
            q_cpu[q_idx + head_dim / 2] = q0 * sin_t + q1 * cos_t;

            let k_idx = head * head_dim + i;
            let k0 = k_cpu[k_idx];
            let k1 = k_cpu[k_idx + head_dim / 2];
            k_cpu[k_idx] = k0 * cos_t - k1 * sin_t;
            k_cpu[k_idx + head_dim / 2] = k0 * sin_t + k1 * cos_t;
        }
    }

    let ctx = get_metal_context().expect("Metal context");
    let q_buf = ctx.prepare_f32_dense_tensor(&q);
    let k_buf = ctx.prepare_f32_dense_tensor(&k);
    ctx.dispatch_rope(&q_buf, &k_buf, pos, head_dim, head_dim, 10000.0, 1.0)
        .expect("GPU dispatch");
    unsafe {
        std::ptr::copy_nonoverlapping(q_buf.contents() as *const f32, q.as_mut_ptr(), dim);
        std::ptr::copy_nonoverlapping(k_buf.contents() as *const f32, k.as_mut_ptr(), dim);
    }

    for i in 0..dim {
        assert!((q[i] - q_cpu[i]).abs() < 1e-5);
        assert!((k[i] - k_cpu[i]).abs() < 1e-5);
    }
}
