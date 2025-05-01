import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run training, prediction, and submission steps.")
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Name of the model to train or test (e.g., UNet, SegNet, LinkNet34, NL_LinkNet_EGaussian)."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to a pre-trained model (.pt file). If provided, training will be skipped."
    )
    parser.add_argument(
        "--test_set", type=str, default="dataset/test_set_images",
        help="Path to the test dataset."
    )
    parser.add_argument(
        "--output_masks", type=str, default="test_set_masks",
        help="Directory to save predicted test masks."
    )
    parser.add_argument(
        "--submission_output", type=str, default="submission.csv",
        help="File path for the final submission."
    )

    args = parser.parse_args()

    # Step 1: Training phase
    if args.model_path:
        print(f"Using pre-trained model: {args.model_path}. Skipping training phase.")
        model_path = args.model_path
        model_flag = None

        # Determine which model flag to use for `test_pred.py`
        if "UNet" in args.model_path:
            model_flag = "--unet"
        elif "SegNet" in args.model_path:
            model_flag = "--segnet"
        elif "LinkNet" in args.model_path:
            model_flag = "--linknet"
        elif "DLinkNet" in args.model_path:
            model_flag = "--dlinknet"

        if not model_flag:
            print("Error: Could not determine the model type from the model path.")
            return
    else:
        if not args.model_name:
            print("Error: Either --model_path or --model_name must be provided.")
            return

        print("Starting training phase...")
        train_command = ["python", "train.py", "--model_name", args.model_name]
        subprocess.run(train_command, check=True)

        model_path = f"models/trained_model_{args.model_name}.pt"
        model_flag = f"--{args.model_name.lower()}"
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path} after training.")
            return

    # Step 2: Testing/prediction phase
    print("Starting prediction phase...")
    if args.test_set:
        test_command = [
        "python", "test_pred.py",
        model_flag,  model_path,
        "--test_set", args.test_set,
    ]
    elif args.output_masks:
        test_command = [
        "python", "test_pred.py",
        model_flag,  model_path,
        "--output_masks", args.output_masks,
    ]
    elif args.output_masks and args.test_set:
        test_command = [
        "python", "test_pred.py",
        model_flag, "True",  # Pass the flag with the value 'True'
        "--test_set", args.test_set,
        "--output_masks", args.output_masks
    ]
    else: 
        test_command = [
            "python", "test_pred.py",
            model_flag,  model_path
        ]
    subprocess.run(test_command, check=True)

    # Step 3: Submission generation phase
    print("Starting submission generation...")
    submission_command = [
        "python", "mask_to_submission.py",
        "--input_masks", args.output_masks,
        "--output_file", args.submission_output
    ]
    subprocess.run(submission_command, check=True)

    print(f"Pipeline completed. Submission saved to {args.submission_output}")


if __name__ == "__main__":
    main()
