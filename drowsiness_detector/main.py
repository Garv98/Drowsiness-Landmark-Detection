import os
import logging
import argparse
from core import (
    setup_kaggle, download_datasets, setup_dirs,
    process_dataset, setup_training_data,
    load_landmarks, load_saved_model, train_model,
    evaluate_model, datasets
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_args():
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness Detection System")
    parser.add_argument('--setup', action='store_true', help='Setup Kaggle API config')
    parser.add_argument('--download_datasets', action='store_true', help='Download datasets from Kaggle')
    parser.add_argument('--train_model', type=str, help='Path to training dataset')
    parser.add_argument('--preprocess_data', action='store_true', help='Whether to preprocess dataset or not')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
    return parser.parse_args()

def main():
    
    if not os.path.exists('Logs'):
        os.makedirs('Logs', exist_ok=True)

    logging.basicConfig(
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join('Logs', 'debug.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('PIL.JpegImagePlugin').setLevel(logging.WARNING)
    logging.getLogger('exifread').setLevel(logging.WARNING)
    
    args = get_args()

    logging.basicConfig(
        encoding='utf-8', level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler("./Logs/debug.log"),
            logging.StreamHandler()
        ]
    )


    categories = ["Fatigue Subjects", "Active Subjects"]

    if args.setup:
        logging.info("Setting up Kaggle")
        setup_kaggle()

    if args.download_datasets:
        logging.info("Downloading datasets from Kaggle")
        download_datasets(datasets)

    if args.train_model:
        logging.info("Preparing training data")
        setup_dirs(categories)
        train_gen, val_gen = setup_training_data(
            data=args.train_model,
            img_size=(145,145),
            batch_size=32,
            validation_split=0.2
        )
        model = load_saved_model()
        history = train_model(model, train_gen, val_gen, epochs=args.epochs)

if __name__ == '__main__':
    main()