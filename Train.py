import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from model.dual_branch_model import DualBranchModelAFE


def create_balanced_test_set(x_1d, x_2d, y):
    """
    Create a balanced test set with an equal number of positive and negative samples.

    :param x_1d: 1D features
    :param x_2d: 2D image features
    :param y: Labels
    :return: Balanced 1D features, 2D image features, and labels
    """
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y == 0).nonzero(as_tuple=True)[0]

    # Ensure an equal number of positive and negative samples
    selected_pos_indices = pos_indices[:8]
    selected_neg_indices = neg_indices[:8]

    test_indices = torch.cat((selected_pos_indices, selected_neg_indices))

    x_test_1d = x_1d[test_indices]
    x_test_2d = x_2d[test_indices]
    y_test = y[test_indices]

    return x_test_1d, x_test_2d, y_test


def train_dual_branch_model(preprocessed_data_path, pretrained_1d_path, pretrained_2d_path, save_path):
    """
    Train the dual-branch model.

    :param preprocessed_data_path: Path to the preprocessed dataset
    :param pretrained_1d_path: Path to the pretrained 1D convolutional weights
    :param pretrained_2d_path: Path to the pretrained 2D convolutional weights
    :param save_path: Path to save the trained model weights
    """
    # Load the saved dataset
    saved_data = torch.load(preprocessed_data_path)
    x_1d = torch.tensor(saved_data['features1d'], dtype=torch.float32).unsqueeze(1)
    x_2d = saved_data['images2d']
    y = torch.tensor(saved_data['labels'], dtype=torch.float32)

    # Split into training and validation sets
    train_size = int(0.8 * len(y))
    x_train_1d, x_val_1d = x_1d[:train_size], x_1d[train_size:]
    x_train_2d, x_val_2d = x_2d[:train_size], x_2d[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create a balanced validation set
    x_val_1d, x_val_2d, y_val = create_balanced_test_set(x_val_1d, x_val_2d, y_val)

    train_dataset = TensorDataset(x_train_1d, x_train_2d, y_train)
    val_dataset = TensorDataset(x_val_1d, x_val_2d, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the model
    model = DualBranchModelAFE()

    # Load pretrained weights
    model.conv1d_branch.load_state_dict(torch.load(pretrained_1d_path), strict=False)
    model.conv2d_branch.load_state_dict(torch.load(pretrained_2d_path), strict=False)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Training process
    num_epochs = 100
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x_1d_batch, x_2d_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_1d_batch, x_2d_batch).squeeze(1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        scheduler.step()

    # Save the trained model weights
    torch.save(model.state_dict(), save_path)

    # Plot the training loss curve
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curve')
    plt.show()

    # Evaluate on the validation set
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_1d_batch, x_2d_batch, y_batch in val_loader:
            outputs = model(x_1d_batch, x_2d_batch).squeeze(1)
            preds = (outputs > 0.5).float()
            all_preds.append(preds)

    all_preds = torch.cat(all_preds)
    print(classification_report(y_val, all_preds))


if __name__ == "__main__":
    # Define paths (relative to the project root)
    preprocessed_data_path = "data/processed_dataset.pt"
    pretrained_1d_path = "weights/trained_1DCNN_weights.pth"
    pretrained_2d_path = "weights/trained_2DCNN_weights.pth"
    save_path = "weights/dual_branch_model_weights.pth"

    # Train the dual-branch model
    train_dual_branch_model(preprocessed_data_path, pretrained_1d_path, pretrained_2d_path, save_path)