from training.train import train_dual_branch_model
from prediction.predict import predict_and_save_results

if __name__ == "__main__":
    print("Choose an action:")
    print("1. Train the model")
    print("2. Predict using the model")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("Starting training...")
        train_dual_branch_model()
    elif choice == "2":
        print("Starting prediction...")
        predict_and_save_results()
    else:
        print("Invalid choice. Please enter 1 for training or 2 for prediction.")
