
"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
from torchvision import transforms
# import data_setup, engine, model_builder, utils
from datasets import data_setup
from models import base_model 
from trainers import engine, train_setup
from utils import utils, experiment_track




# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for num_epochs
parser.add_argument("--model_name",
                    default="Not Defined",
                    type=str,
                    help="Name of the model")

parser.add_argument("--experiment_name",
                    default="first_experiment",
                    type=str,
                    help="Name of the experiment")

parser.add_argument("--epochs", 
                     default=10, 
                     type=int, 
                     help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of hidden units in hidden layers")

# Get an arg for learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate to use for model")

# Create an arg for training directory 
parser.add_argument("--train_dir",
                    default="datasets/sample_data/pizza_steak_shushi/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Create an arg for test directory 
parser.add_argument("--test_dir",
                    default="datasets/sample_data/pizza_steak_shushi/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")

# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
MODEL_NAME = args.model_name
EXPERIMENT_NAME = args.experiment_name
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_size
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = base_model.BiRNN(
    input_size=5,
    hidden_size=HIDDEN_UNITS,
    num_layers= 128,
    output_shape=2
).to(device)

# BiRNN(input_size, hidden_size, num_layers, num_classes)
# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
train_setup.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             writer=experiment_track.create_writer(experiment_name=EXPERIMENT_NAME,
                                       model_name=MODEL_NAME,
                                       extra=f"{NUM_EPOCHS}_epochs"))

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="saved/",
                 model_name=f"{MODEL_NAME}.pth")


