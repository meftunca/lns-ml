use repugnant_pickle::torch::RepugnantTorchTensors;

fn main() -> anyhow::Result<()> {
    let path = "/Users/mapletechnologies/Desktop/big_projects/lns-ml/models/tinyllama1.1B/pytorch_model.bin";
    let tensors = RepugnantTorchTensors::new_from_file(path).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    if let Some(t) = tensors.iter().next() {
        println!("{:#?}", t);
    }
    Ok(())
}
