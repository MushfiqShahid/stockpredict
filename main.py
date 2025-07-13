import subprocess
import sys
import os

print("Which company's stock do you want to predict?")
print("Press 1 for Alphabet (Google)")
print("Press 2 for Amazon")
print("Press 3 for AMD")
print("Press 4 for Apple")
print("Press 5 for Intel")
print("Press 6 for Meta (Facebook)")
print("Press 7 for Microsoft")
print("Press 8 for Netflix")
print("Press 9 for Nvidia")
print("Press 10 for Tesla")

choice = input("Enter your choice (1-10): ")

# Path to model scripts
model_scripts = {
    '1': os.path.join("models", "alphabet_lstm_forecasting.py"),
    '2': os.path.join("models", "amazon_lstm_forecasting.py"),
    '3': os.path.join("models", "amd_lstm_forecasting.py"),
    '4': os.path.join("models", "apple_lstm_forecasting.py"),
    '5': os.path.join("models", "intel_lstm_forecasting.py"),
    '6': os.path.join("models", "meta_lstm_forecasting.py"),
    '7': os.path.join("models", "microsoft_lstm_forecasting.py"),
    '8': os.path.join("models", "netflix_lstm_forecasting.py"),
    '9': os.path.join("models", "nvidia_lstm_forecasting.py"),
    '10': os.path.join("models", "tesla_lstm_forecasting.py")
}

script_to_run = model_scripts.get(choice)

if script_to_run:
    subprocess.run([sys.executable, script_to_run])
else:
    print("Invalid choice. Please enter a number between 1 and 10.")