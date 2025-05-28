import subprocess
import sys
import os

print("Which company's stock do you want to predict?")
print("Press 1 for Apple")
print("Press 2 for Microsoft")
print("Press 3 for Tesla")

choice = input("Enter your choice (1, 2, or 3): ")

# Path to model scripts
model_scripts = {
    '1': os.path.join("models", "apple_lstm_forecasting.py"),
    '2': os.path.join("models", "microsoft_lstm_forecasting.py"),
    '3': os.path.join("models", "tesla_lstm_forecasting.py")
}

script_to_run = model_scripts.get(choice)

if script_to_run:
    subprocess.run([sys.executable, script_to_run])
else:
    print("Invalid choice. Please enter 1, 2, or 3.")
