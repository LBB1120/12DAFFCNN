# This is the entry point script for the Dual-Branch CNN Project
# Import necessary modules from the separated structure

from training.train import train_dual_branch_model
from prediction.predict import predict_and_save_results

if __name__ == '__main__':
    # Train the model
    train_dual_branch_model()
    # Predict and save results
    predict_and_save_results()
