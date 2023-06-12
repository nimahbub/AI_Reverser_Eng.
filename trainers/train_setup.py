from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from trainers import engine
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = engine.train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = engine.test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
      if writer:
          # Add results to SummaryWriter
          writer.add_scalars(main_tag="Loss", 
                              tag_scalar_dict={"train_loss": train_loss,
                                              "test_loss": test_loss},
                              global_step=epoch)
          writer.add_scalars(main_tag="Accuracy", 
                              tag_scalar_dict={"train_acc": train_acc,
                                              "test_acc": test_acc}, 
                              global_step=epoch)

          # Close the writer
          writer.close()
      else:
          pass
  ### End new ###





  # Visualization of loss and accuracy curve 
  trn_loss = results["train_loss"]
  tst_loss = results["test_loss"]

  trn_acc = results["train_acc"]
  tst_acc = results["test_acc"]

  epos = range(len(results["train_loss"]))


  plt.style.use('ggplot')
  # Turn interactive plotting off
  plt.ioff()

  # Create a new figure, plot into it, then close it so it never gets displayed
  fig = plt.figure(figsize=(8,5))
  plt.plot(epos, trn_loss, "bo-",label="train_loss")
  plt.plot(epos, tst_loss,"ro-", label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.savefig('figures/loss_curve')
  plt.close(fig)


  # Create a new figure, plot into it, then close it so it never gets displayed
  fig = plt.figure(figsize=(8,5))
  plt.plot(epos, trn_acc, "bo-",label="train_accuracy")
  plt.plot(epos, tst_acc, "ro-",label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()
  plt.savefig('figures/accuracy_curve')
  plt.close(fig)


  # Return the filled results at the end of the epochs
  return results


