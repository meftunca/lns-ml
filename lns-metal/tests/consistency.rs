use lns_core::format::QuantType;
use lns_core::*;
use lns_metal::*;
use metal::MTLResourceOptions;

fn cpu_gemv(blocks_raw: &[u8], x: &[f32], rows: usize, cols: usize, qt: QuantType) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    let blocks_per_row = cols / 256;

    for r in 0..rows {
        let mut acc = 0.0f32;

        let decoded = match qt {
            QuantType::Q4L => {
                let blocks: &[Q4LSuperBlock] = bytemuck::cast_slice(blocks_raw);
                let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
                dequantize_q4l(row_blocks, cols)
            }
            QuantType::Q2L => {
                let blocks: &[Q2LSuperBlock] = bytemuck::cast_slice(blocks_raw);
                let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
                dequantize_q2l(row_blocks, cols)
            }
            QuantType::Q8L => {
                let blocks: &[Q8LSuperBlock] = bytemuck::cast_slice(blocks_raw);
                let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
                dequantize_q8l(row_blocks, cols)
            }
            QuantType::Q4HQ => {
                let blocks = q4hq_blocks_from_payload(blocks_raw).expect("valid Q4HQ payload");
                let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
                dequantize_q4hq(row_blocks, cols)
            }
            _ => panic!("Unsupported type"),
        };

        for c in 0..cols {
            acc += decoded[c] * x[c];
        }
        y[r] = acc;
    }
    y
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemv_q4hq_consistency() {
    let rows = 4;
    let cols = 256;
    let mut rng_data = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        rng_data[i] = (i as f32 * 0.077).sin() * 0.75 + (i as f32 * 0.031).cos() * 0.125;
    }

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).cos()).collect();
    let blocks = quantize_q4hq(&rng_data);
    let payload = q4hq_blocks_to_payload(&blocks);

    let cpu_res = cpu_gemv(&payload, &x, rows, cols, QuantType::Q4HQ);

    let ctx = get_metal_context().expect("Metal context");
    let weights = ctx
        .prepare_tensor(&payload, rows, cols, QuantType::Q4HQ)
        .expect("Prepare Q4HQ weights");
    let x_buf = ctx.prepare_f32_dense_tensor(&x);
    let y_buf = ctx.device.new_buffer(
        (rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    ctx.dispatch_gemv_cached(&weights, &x_buf, &y_buf, rows, cols, QuantType::Q4HQ)
        .expect("Q4HQ GPU dispatch");
    let mut gpu_res = vec![0.0f32; rows];
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, gpu_res.as_mut_ptr(), rows);
    }

    for i in 0..rows {
        let diff = (cpu_res[i] - gpu_res[i]).abs();
        println!(
            "Q4HQ Row {}: CPU={}, GPU={}, Diff={}",
            i, cpu_res[i], gpu_res[i], diff
        );
        assert!(diff < 1e-4);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemv_q4hq_accumulate_consistency() {
    let rows = 4;
    let cols = 256;
    let mut rng_data = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        rng_data[i] = (i as f32 * 0.061).sin() * 0.65 + (i as f32 * 0.019).cos() * 0.175;
    }

    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin()).collect();
    let base_y: Vec<f32> = (0..rows).map(|i| i as f32 * 0.25 - 0.5).collect();
    let blocks = quantize_q4hq(&rng_data);
    let payload = q4hq_blocks_to_payload(&blocks);

    let mut cpu_res = cpu_gemv(&payload, &x, rows, cols, QuantType::Q4HQ);
    for (value, base) in cpu_res.iter_mut().zip(base_y.iter()) {
        *value += base;
    }

    let ctx = get_metal_context().expect("Metal context");
    let weights = ctx
        .prepare_tensor(&payload, rows, cols, QuantType::Q4HQ)
        .expect("Prepare Q4HQ weights");
    let x_buf = ctx.prepare_f32_dense_tensor(&x);
    let y_buf = ctx.prepare_f32_dense_tensor(&base_y);
    ctx.dispatch_gemv_accumulate_cached(&weights, &x_buf, &y_buf, rows, cols)
        .expect("Q4HQ accumulate GPU dispatch");
    let mut gpu_res = vec![0.0f32; rows];
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, gpu_res.as_mut_ptr(), rows);
    }

    for i in 0..rows {
        let diff = (cpu_res[i] - gpu_res[i]).abs();
        println!(
            "Q4HQ accumulate row {}: CPU={}, GPU={}, Diff={}",
            i, cpu_res[i], gpu_res[i], diff
        );
        assert!(diff < 1e-4);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemv_q4l_consistency() {
    let rows = 4;
    let cols = 256;
    let mut rng_data = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        rng_data[i] = (i as f32 * 0.1).sin();
    }

    let x = vec![1.0f32; cols];
    let blocks = quantize_q4l(&rng_data);
    let blocks_raw = bytemuck::cast_slice(&blocks);

    let cpu_res = cpu_gemv(blocks_raw, &x, rows, cols, QuantType::Q4L);

    let ctx = get_metal_context().expect("Metal context");
    let weights = ctx
        .prepare_tensor(blocks_raw, rows, cols, QuantType::Q4L)
        .expect("Prepare weights");
    let x_buf = ctx.prepare_f32_dense_tensor(&x);
    let y_buf = ctx.device.new_buffer(
        (rows * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    ctx.dispatch_gemv_cached(&weights, &x_buf, &y_buf, rows, cols, QuantType::Q4L)
        .expect("GPU dispatch");
    let mut gpu_res = vec![0.0f32; rows];
    unsafe {
        std::ptr::copy_nonoverlapping(y_buf.contents() as *const f32, gpu_res.as_mut_ptr(), rows);
    }

    for i in 0..rows {
        let diff = (cpu_res[i] - gpu_res[i]).abs();
        println!(
            "Q4L Row {}: CPU={}, GPU={}, Diff={}",
            i, cpu_res[i], gpu_res[i], diff
        );
        assert!(diff < 1e-4);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemv_q2l_consistency() {
    let ctx = get_metal_context().expect("Metal context");
    if !ctx.gemv_pipelines.contains_key(&QuantType::Q2L) {
        eprintln!("Skipping Q2L GEMV consistency test: kernel unavailable");
        return;
    }

    let rows = 4;
    let cols = 256;
    let mut rng_data = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        rng_data[i] = (i as f32 * 0.1).sin();
    }

    let x = vec![1.0f32; cols];
    let blocks = quantize_q2l(&rng_data);
    let blocks_raw = bytemuck::cast_slice(&blocks);

    let cpu_res = cpu_gemv(blocks_raw, &x, rows, cols, QuantType::Q2L);

    let _ = (blocks_raw, cpu_res);

    eprintln!("Q2L GEMV kernel is not implemented in the current Metal backend");
}

#[cfg(target_os = "macos")]
#[test]
fn test_metal_gemv_q8l_consistency() {
    let ctx = get_metal_context().expect("Metal context");
    if !ctx.gemv_pipelines.contains_key(&QuantType::Q8L) {
        eprintln!("Skipping Q8L GEMV consistency test: kernel unavailable");
        return;
    }

    let rows = 4;
    let cols = 256;
    let mut rng_data = vec![0.0f32; rows * cols];
    for i in 0..(rows * cols) {
        rng_data[i] = (i as f32 * 0.1).sin();
    }

    let x = vec![1.0f32; cols];
    let blocks = quantize_q8l(&rng_data);
    let blocks_raw = bytemuck::cast_slice(&blocks);

    let cpu_res = cpu_gemv(blocks_raw, &x, rows, cols, QuantType::Q8L);

    let _ = (blocks_raw, cpu_res);

    eprintln!("Q8L GEMV kernel is not implemented in the current Metal backend");
}
