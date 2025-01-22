import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data_utils import PredictionDataset
from model import DualBranchModelAFE


def predict_and_save_results():
    """
    Perform inference using the dual-branch model and save the results to an Excel file.
    """
    # Path to the prediction dataset (relative path)
    prediction_data_path = './data/prediction/1-2D_final_dataset.npy'

    # Load prediction data
    prediction_data = np.load(prediction_data_path, allow_pickle=True)

    # Initialize the model
    model = DualBranchModelAFE()

    # Load pretrained weights (relative path)
    model_weights_path = './weights/dual_branch_fusion_weights.pth'
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()  # Set the model to evaluation mode

    # Create a DataLoader for the prediction data
    prediction_dataset = PredictionDataset(prediction_data)
    prediction_loader = DataLoader(prediction_dataset, batch_size=32, shuffle=False)

    # Perform predictions
    predictions = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for features_1d, image_data_2d, _ in prediction_loader:  # Coordinates are not needed
            output = model(features_1d, image_data_2d).squeeze(1)  # Get prediction probabilities
            predictions.extend(output.tolist())

    # Save predictions to an Excel file (relative path)
    output_path = './results/prediction_results.xlsx'
    df = pd.DataFrame({'Prediction Probability': predictions})
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}.")


if __name__ == '__main__':
    predict_and_save_results()