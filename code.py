import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from faker import Faker
import os # Import os for better path handling if needed, but not strictly necessary for this fix
 
# --- Data Generation and Embedding Functions ---
 
def count_letters(name):
  """
  Counts the frequency of each letter in a name and returns a list of 26 integers.
  """
  counts = [0] * 27
  lower_name = name.lower()
 
  # Use string.ascii_lowercase for efficiency and clarity
  for i, letter in enumerate(string.ascii_lowercase):
    # This is slightly inefficient as it scans the whole string 26 times.
    # A more efficient approach is to use a dictionary or Counter,
    # but for a small name string, this is acceptable for simplicity.
    letter_count = lower_name.count(letter)
    counts[i] = letter_count
  last_char = name[-1].lower()
  if last_char == 'i' or last_char == 'a':
    counts[26] = 1
  else:
    counts[26] = 0  
 
  return counts
 
def nameEmbedding(name, intGender):
  letter_counts = count_letters(name)
  # Append the gender integer (0.9 for male, 0.1 for female)
  letter_counts.append(intGender)
  return letter_counts
 
# --- Data Generation and CSV Creation ---
 
# Initialize Faker with the Indian English locale (en_IN)
fake = Faker('en_IN')
 
lstnameEmbedding = []
CSV_FILENAME = 'nameEmbedding.csv' # Define a constant for the filename
 
# Generate multiple Indian male names (Gender label 0.9)
for _ in range(500):
  lstnameEmbedding.append(nameEmbedding(fake.first_name_male(), 0))
 
# Generate multiple Indian female names (Gender label 0.1)
for _ in range(500):
  lstnameEmbedding.append(nameEmbedding(fake.first_name_female(), 1))
 
df = pd.DataFrame(lstnameEmbedding)
 
# Save the CSV in the current working directory
df.to_csv(CSV_FILENAME, index=False)
print(f"Data saved to {CSV_FILENAME}")
 
# --- PyTorch Model Setup and Training ---
 
 
try:
  dataset = np.genfromtxt(CSV_FILENAME, delimiter=',', skip_header=1)
except FileNotFoundError:
  print(f"\nERROR: Could not find '{CSV_FILENAME}'. Ensure the script has write/read permissions.")
  exit()
 
X = dataset[:, 0:27]
y = dataset[:, 27]
 
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
 
# define the model
model = nn.Sequential(
    nn.Linear(27, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
 
print("\nModel Structure:")
print(model)
 
# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 300
batch_size = 10
 
print("\nStarting Training (100 Epochs):")
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(f'Finished epoch {epoch}, latest loss {loss.item():.4f}') # Using .item() for clean printing
 
# --- Prediction Function ---
 
def predictGenderFromName(strName):
  # Get letter counts, convert to tensor, and pass through model
  letter_counts_np = np.array(count_letters(strName), dtype=np.float32)
  y_pred = model(torch.tensor(letter_counts_np, dtype=torch.float32))
 
  # The output is a tensor, convert to float for comparison
  prediction_value = y_pred.item()
  print(y_pred)
  if prediction_value > 0.5:
    gender = 'Female'
  else:
    gender = 'Male'
   
  # print(f"Name: {strName:<12} -> Prediction: {gender:<6} (Score: {prediction_value:.4f})")
  print(f"Name: {strName} -> Prediction: {gender}")
 
 
# --- Test Predictions (100 Epochs) ---
print("\nPredictions (100 Epochs):")
predictGenderFromName('Kamlesh')
predictGenderFromName('Vidhya')
predictGenderFromName('Paramita')
predictGenderFromName('Mounika')
predictGenderFromName('Ashwin')
predictGenderFromName('Balachandar')
predictGenderFromName('Sunil')
 
print('if Wrong, Increase epoch to 1000 and see results')
 
predictGenderFromName('Abhinav')
predictGenderFromName('Astha')
predictGenderFromName('Abel')
predictGenderFromName('Thanmai')
predictGenderFromName('Maihar')
 