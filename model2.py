import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the preprocessed winter data (assuming the file is correctly loaded in the environment)
data = pd.read_csv('processed_snotel_data_filled 1.csv')
data['date'] = pd.to_datetime(data['date'])

# Define features (X) and target variable (y)
X = data.drop(columns=['SWE', 'date'])  # Drop the target variable and Date
y = data['SWE']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for LSTM (samples, time_steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # 1 time step per sample

# Split the data into 80% training and 20% temporary (for test and validation)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Further split the temporary data into 50% test and 50% validation (10% test, 10% validation)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(Dense(1))  # Output layer with 1 neuron for SWE prediction

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Model Loss: {loss}")

# Predict SWE for the test set
y_pred_lstm = lstm_model.predict(X_test)

# Calculate R-squared for evaluation
r2 = r2_score(y_test, y_pred_lstm)
print(f"LSTM R-squared: {r2}")

for sample_index in range(5):
    print(f"\nTesting for sample index: {sample_index}")
    print("Input features (X_test) for the selected sample:")
    print(X_test[sample_index])
    print(f"\nActual SWE value for this sample: {y_test.iloc[sample_index]}")

    # Predict SWE for this sample
    predicted_swe = lstm_model.predict(X_test[sample_index].reshape(1, 1, X_test.shape[2]))
    print(f"\nPredicted SWE value for this sample: {predicted_swe[0][0]}")


'''# Example: Predict SWE for a new sample
sample_index = 10
print("Input features (X_test) for the selected sample:")
print(X_test[sample_index])

print(f"\nActual SWE value for this sample: {y_test.iloc[sample_index]}")

# Predict SWE for this sample
predicted_swe = lstm_model.predict(X_test[sample_index].reshape(1, 1, X_test.shape[2]))

# Print the predicted SWE value
print(f"\nPredicted SWE value for this sample: {predicted_swe[0][0]}")'''




# Calculate the Mean Absolute Percentage Error (MAPE)
y_test_flat = y_test.values.flatten()  # Flatten if it's a Pandas series
y_pred_lstm_flat = y_pred_lstm.flatten()  # Flatten if it's a numpy array

# Filter out instances where actual SWE is zero (to prevent division by zero in MAPE)
non_zero_indices = y_test_flat != 0
y_test_filtered = y_test_flat[non_zero_indices]
y_pred_filtered = y_pred_lstm_flat[non_zero_indices]

# Calculate the MAPE for the filtered data
mape = np.mean(np.abs((y_pred_filtered - y_test_filtered) / y_test_filtered)) * 100

 
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Optionally: Plot the training and validation loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training vs Validation Loss (LSTM)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
