use lns_inference::Tokenizer;
fn main() {
    let t = Tokenizer::from_file(
        "/Users/mapletechnologies/Desktop/big_projects/lns-ml/models/tinyllama1.1B/tokenizer.json",
    )
    .unwrap();
    for i in 0..100 {
        println!("{}: {:?}", i, t.decode(i));
    }
}
