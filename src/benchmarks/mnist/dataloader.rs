use std::fs::File;
use std::io::{self, Read};

pub struct MNISTDataloader {
    training_images_filepath: String,
    training_labels_filepath: String,
    test_images_filepath: String,
    test_labels_filepath: String,
}

impl MNISTDataloader {
    pub fn new(
        training_images_filepath: String,
        training_labels_filepath: String,
        test_images_filepath: String,
        test_labels_filepath: String,
    ) -> Self {
        MNISTDataloader {
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
        }
    }

    fn load_labels(&self, filepath: &str) -> io::Result<Vec<u8>> {
        let mut file = File::open(filepath)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 2049 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2049, got {}", magic),
            ));
        }

        Ok(buffer[8..].to_vec())
    }

    fn load_images(&self, filepath: &str) -> io::Result<Vec<Vec<Vec<u8>>>> {
        let mut file = File::open(filepath)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 2051 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2051, got {}", magic),
            ));
        }

        let size = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
        let rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as usize;
        let cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as usize;

        let mut images = Vec::with_capacity(size);
        let mut index = 16;
        for _ in 0..size {
            let mut image = Vec::with_capacity(rows);
            for _ in 0..rows {
                let row = buffer[index..index + cols].to_vec();
                index += cols;
                image.push(row);
            }
            images.push(image);
        }

        Ok(images)
    }

    fn load_dataset(&self, images_filepath: &str, labels_filepath: &str) -> io::Result<(Vec<Vec<Vec<u8>>>, Vec<u8>)> {
        let labels = self.load_labels(labels_filepath)?;
        let images = self.load_images(images_filepath)?;
        Ok((images, labels))
    }

    pub fn load_all(&self) -> io::Result<(Dataset, Dataset)> {
        let train = self.load_dataset(&self.training_images_filepath, &self.training_labels_filepath)?;
        let test = self.load_dataset(&self.test_images_filepath, &self.test_labels_filepath)?;
        Ok((train, test))
    }
}

type Dataset = (Vec<Vec<Vec<u8>>>, Vec<u8>);
