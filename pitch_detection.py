import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Function to extract chroma features from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    aggregated_chroma = np.mean(chroma, axis=1)  # Calculate the mean along the columns
    return aggregated_chroma

# Example dataset preparation
in_tune_features = extract_features('wav/intune1.wav')
sharp_features1 = extract_features('wav/intune2.wav')
sharp_features2 = extract_features('wav/intune4.wav')
flat_features1 = extract_features('wav/intune3.wav')
flat_features2 = extract_features('wav/intune5.wav')

# Combine features and assign labels (1 for in-tune, 0 for out-of-tune)
X = np.vstack((in_tune_features, sharp_features1, sharp_features2, flat_features1, flat_features2))
y = np.hstack((np.ones(len(in_tune_features)), np.zeros(len(sharp_features1) + len(sharp_features2) + len(flat_features1) + len(flat_features2))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM classifier with scaled features
svm = SVC(kernel='linear')
svm.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = svm.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy}")

# Real-time analysis (Example: Using a single user input file for demonstration)
user_input_features = extract_features('wav/violin_thing.wav')  # Extract features from user input
user_input_features_scaled = scaler.transform(user_input_features.reshape(1, -1))  # Scale user input features

prediction = svm.predict(user_input_features_scaled)

# Display prediction result to the user
if prediction == 1:
    print("The note played is in tune.")
else:
    print("The note played is not in tune. It is sharp or flat.")
