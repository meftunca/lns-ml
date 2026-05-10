fn main() {
    let path =
        "/Users/mapletechnologies/Desktop/big_projects/lns-ml/models/tinyllama1.1B/tinyllama.lns";
    let bytes = std::fs::read(path).unwrap();
    let model = unsafe { rkyv::archived_root::<lns_core::format::LnsModel>(&bytes) };

    for (i, tensor) in model.tensors.iter().enumerate() {
        if i > 5 {
            break;
        } // Just check a few
        let data = tensor.to_f32().unwrap();
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &v in &data {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / data.len() as f32;
        let var = sum_sq / data.len() as f32 - mean * mean;
        println!(
            "Tensor: {}, Mean: {}, Var: {}",
            tensor.name.as_str(),
            mean,
            var
        );
    }
}
