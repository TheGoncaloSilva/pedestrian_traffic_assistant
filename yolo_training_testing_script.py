import sys
from ultralytics import YOLO
import matplotlib.pyplot as plt
import argparse
import logging
import cv2
import os
import random
import pandas as pd
import matplotlib.image as mpimg
import seaborn as sns
import torch
import time

sns.set_style('darkgrid')

def check_data_paths(train_images: str, train_labels: str, test_images: str, test_labels: str, val_images:str, val_labels: str) -> bool:
    """
    Check if the data paths exist
    Attributes:
        train_images: str: Path to the training images
        train_labels: str: Path to the training labels
        test_images: str: Path to the test images
        test_labels: str: Path to the test labels
        val_images: str: Path to the validation images
        val_labels: str: Path to the validation labels
    Returns:
        bool: True if all paths exist
    Raises:
        FileNotFoundError: If any of the paths do not exist
    """
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"Training images path does not exist: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels path does not exist: {train_labels}")
    if not os.path.exists(test_images):
        raise FileNotFoundError(f"Test images path does not exist: {test_images}")
    if not os.path.exists(test_labels):
        raise FileNotFoundError(f"Test labels path does not exist: {test_labels}")
    if not os.path.exists(val_images):
        raise FileNotFoundError(f"Validation images path does not exist: {val_images}")
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels path does not exist: {val_labels}")
    return True

def get_images_size(train_images_path: str) -> list[int, int, int]:
    """
    Get the size of the images
    Attributes:
        train_images_path: str: Path to the training images
    Returns:
        list[int, int, int]: Height, Width and Channels of the image
    """
    # get first image name of train_images_path
    image_name = os.listdir(train_images_path)[0]
    image = cv2.imread(train_images + "/" + image_name)

    # Get the size of the image
    height, width, channels = image.shape
    return [height, width, channels]

def print_settings() -> None:
    logging.info("Settings:")
    logging.info(f" - Train images path: {train_images}")
    logging.info(f" - Train labels path: {train_labels}")
    logging.info(f" - Test images path: {test_images}")
    logging.info(f" - Test labels path: {test_labels}")
    logging.info(f" - Validation images path: {val_images}")
    logging.info(f" - Validation labels path: {val_labels}")
    logging.info(f" - Model name: {modelName}")
    logging.info(f" - Epochs: {args.epochs}")
    logging.info(f" - Image size: {image_info[0]}x{image_info[1]}")
    logging.info(f" - Image Channels: {image_info[2]}")
    logging.info(f" - Device: {device}")
    logging.info(f" - Patience: {args.patience}")
    logging.info(f" - Batch size: {args.batch}")
    logging.info(f" - Optimizer: {args.optimizer}")
    logging.info(f" - Initial Learning rate: {args.lr0}")
    logging.info(f" - Weight decay: {args.weight_decay}")
    logging.info(f" - Seed: {args.seed}")


def print_execution_time(start_time: float, end_time: float, designation: str) -> None:
    """
    Print the execution time
    Attributes:
        start_time: float: Start time of
        end_time: float: End time of the execution
    """
    execution_time: float = end_time - start_time
    days, remainder = divmod(execution_time, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        logging.info("%s took: %d days %02d:%02d:%02d", designation, int(days), int(hours), int(minutes), int(seconds))
    else:
        logging.info("%s took: %02d:%02d:%02d", designation, int(hours), int(minutes), int(seconds))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="Log threshold (default=DEBUG)", type=str, default='DEBUG')
    parser.add_argument("--logFile", help="Log file (default=logs.log)", type=str, default='logs.txt')
    parser.add_argument("--dataset", help="Dataset to use (crossroads, mixed or traffic_lights)", type=str, default='mixed')
    parser.add_argument("--model", help="Path to the config file (default=yolo11n)", type=str, default='yolo11n')
    parser.add_argument("--epochs", help="Number of epochs (default=300)", type=int, default=300)
    parser.add_argument("--batch", help="Batch size (default=16)", type=int, default=16)
    parser.add_argument("--patience", help="Early stopping patience (default=100)", type=int, default=100)
    parser.add_argument("--seed", help="Random seed", type=int, default=0)
    parser.add_argument("--optimizer", help="Optimizer to use (default=auto)", type=str, default='auto')
    parser.add_argument("--lr0", help="Initial Learning rate (default=0.01)", type=float, default=0.01)
    parser.add_argument("--weight_decay", help="Weight decay (default=0.0005)", type=float, default=0.0005)
    args = parser.parse_args()

    numericLogLevel = getattr(logging, args.log.upper(), None)
    if not isinstance(numericLogLevel, int):
        raise ValueError('Invalid log level: %s' % numericLogLevel)

    logging.basicConfig(
        level=numericLogLevel, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # configure file logger
    fileLogger = logging.FileHandler(args.logFile)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fileLogger)

    # Set the device to the appropriate one
    deviceType: str = ""
    if torch.cuda.is_available():
        torch.device('cuda')
        device = torch.device("cuda")
    # Check if MPS is available and built/activated
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print (x)
    else:
        logging.info("Device name: CPU")
        device = torch.device("cpu")

    # Configure the data paths
    if args.dataset not in ["crossroads", "mixed", "traffic_lights"]:
        logging.error("Invalid dataset. Please use one of the following: crossroads, mixed, traffic_lights")
        sys.exit(1)
    dataset: str = os.path.join(args.dataset, "dataset", "yolo_format", "data.yaml")
    train_images: str = os.path.join(args.dataset, "dataset", "yolo_format", "train", "images")
    train_labels: str = os.path.join(args.dataset, "dataset", "yolo_format", "train", "labels")
    val_images: str = os.path.join(args.dataset, "dataset", "yolo_format", "valid", "images")
    val_labels: str = os.path.join(args.dataset, "dataset", "yolo_format", "valid", "labels")
    test_images: str = os.path.join(args.dataset, "dataset", "yolo_format", "test", "images")
    test_labels: str = os.path.join(args.dataset, "dataset", "yolo_format", "test", "labels")
    results_dir: str = os.path.join(
        args.dataset, 
        "results", 
        f"{args.model}_{args.epochs}_{args.patience}_{args.batch}_{args.optimizer}_{args.lr0}_{args.weight_decay}_{args.seed}"
        )
    if os.path.exists(results_dir):
        logging.error(f"An execution with the same parameters already exists. Please change the parameters.")
        sys.exit(1)
    os.makedirs(results_dir)
    check_data_paths(train_images, train_labels, test_images, test_labels, val_images, val_labels)

    # Load a sample image to get the size -> image_info[0] = height, image_info[1] = width, image_info[2] = channels
    image_info: list[int, int, int] = get_images_size(train_images)

    # Check the name of the model
    modelName: str = args.model
    if not modelName.endswith('.pt'):
        modelName += '.pt'

    # Loading a pretrained model
    model = YOLO(modelName)

    # free up GPU memory
    torch.cuda.empty_cache()

    # print all configuration settings
    print_settings()

    # Training the model
    logging.info("Starting model training")
    startTime = time.time()
    model.train(
        data=dataset,
        epochs=args.epochs,
        imgsz=(image_info[0], image_info[1], image_info[2]),
        batch=args.batch,
        patience=args.patience,
        optimizer=args.optimizer,
        seed=args.seed,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        project=results_dir
    ) 
    print_execution_time(startTime, time.time(), "Training")

    # print again configuration settings
    print_settings()

    # Model Validation
    logging.info("Starting model Validation")
    startTime = time.time()
    model.val(
        data=dataset, 
        batch=args.batch, 
        imgsz=(image_info[0], image_info[1], image_info[2]), 
        project=results_dir,
        split='test'
        )
    print_execution_time(startTime, time.time(), "Validation")


    
    