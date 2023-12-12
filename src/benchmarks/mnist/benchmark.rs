use std::path::PathBuf;

use crate::models::SimpleNeuralNet;

use super::MNISTDataloader;

pub struct MNISTBenchmark {
    neural_net: SimpleNeuralNet,
    data_loader: MNISTDataloader,
}

impl MNISTBenchmark {
    pub fn new(neural_net: SimpleNeuralNet) -> Self {
        let data_folder = PathBuf::from("src/benchmarks/mnist/data/");
        let data_loader = MNISTDataloader::new(
            data_folder.join("train/images").to_str().unwrap().to_string(),
            data_folder.join("train/labels").to_str().unwrap().to_string(),
            data_folder.join("test/images").to_str().unwrap().to_string(),
            data_folder.join("test/labels").to_str().unwrap().to_string(),
        );

        MNISTBenchmark {
            neural_net,
            data_loader,
        }
    }

    fn normalize_image(&self, image: &[Vec<u8>]) -> Vec<f64> {
        image.iter().flatten().map(|&p| p as f64 / 255.0).collect()
    }

    fn normalize_label(&self, label: u8) -> Vec<f64> {
        let mut expected_output = vec![0.0; self.neural_net.output_nodes];
        expected_output[label as usize] = 1.0;
        expected_output
    }

    pub fn train(&mut self, epoch_count: usize) {
        if let Ok((train_data, _)) = self.data_loader.load_all() {
            let (train_images, train_labels) = train_data;
            for epoch in 1..=epoch_count {
                for (i, image) in train_images.iter().enumerate() {
                    println!("{i}");
                    let label = train_labels[i];
                    let input = self.normalize_image(image);
                    let output = self.normalize_label(label);

                    self.neural_net.train(&input, &output);
                }
                println!("Epoch {} complete.", epoch);
            }
        }
    }

    pub fn test(&self) -> f64 {
        if let Ok((_, test_data)) = self.data_loader.load_all() {
            let (test_images, test_labels) = test_data;
            let mut correct_count = 0;

            for (i, image) in test_images.iter().enumerate() {
                let label = test_labels[i];
                let input = self.normalize_image(image);
                let output = self.neural_net.feed_forward(&input);
                let predicted = output.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();

                if predicted == label as usize {
                    correct_count += 1;
                }
            }

            correct_count as f64 / test_images.len() as f64
        } else {
            0.0
        }
    }
}
