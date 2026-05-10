use lns_core::*;

fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).powi(2);
    }
    (sum / a.len() as f32).sqrt()
}

#[test]
fn test_q4l_roundtrip() {
    let original = vec![0.5, -1.2, 0.0, 3.4, -0.01, 10.5, -5.0, 0.1];
    // Pad to super-block size
    let mut padded = vec![0.0f32; 256];
    padded[..8].copy_from_slice(&original);

    let blocks = quantize_q4l(&padded);
    let decoded = dequantize_q4l(&blocks, 256);

    let rmse = calculate_rmse(&padded, &decoded);
    println!("Q4L RMSE: {}", rmse);
    assert!(rmse < 0.1); // LNS should be reasonably accurate
}

#[test]
fn test_q2l_roundtrip() {
    let original = vec![0.5, -1.2, 0.0, 3.4, -0.01, 10.5, -5.0, 0.1];
    let mut padded = vec![0.0f32; 256];
    padded[..8].copy_from_slice(&original);

    let blocks = quantize_q2l(&padded);
    let decoded = dequantize_q2l(&blocks, 256);

    let rmse = calculate_rmse(&padded, &decoded);
    println!("Q2L RMSE: {}", rmse);
    assert!(rmse < 0.5); // Q2 is much rougher
}

#[test]
fn test_q8l_roundtrip() {
    let original = vec![0.5, -1.2, 0.0, 3.4, -0.01, 10.5, -5.0, 0.1];
    let mut padded = vec![0.0f32; 256];
    padded[..8].copy_from_slice(&original);

    let blocks = quantize_q8l(&padded);
    let decoded = dequantize_q8l(&blocks, 256);

    let rmse = calculate_rmse(&padded, &decoded);
    println!("Q8L RMSE: {}", rmse);
    assert!(rmse < 0.02); // Q8 should be very accurate
}
