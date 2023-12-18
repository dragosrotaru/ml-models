use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
    let model_options = ModelOptions::default();

    let llama = LLama::new("../../mixtral-8x7b-32kseqlen.pth".into(), &model_options).unwrap();

    let predict_options = PredictOptions {
        token_callback: Some(Box::new(|token| {
            println!("token1: {}", token);

            true
        })),
        ..Default::default()
    };

    llama
        .predict(
            "what are the national animals of india".into(),
            predict_options,
        )
        .unwrap();
}
