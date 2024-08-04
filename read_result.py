import joblib
import os

output_pth = os.path.join(".", "output")

# Assuming output_pth is defined
output_file = os.path.join(output_pth, "wham_output.pkl")
loaded_results = joblib.load(output_file)

print(loaded_results)
