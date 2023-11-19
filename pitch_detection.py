import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import soundfile as sf

# Function to extract aggregated chroma features from audio file
def extract_aggregated_features(file_path):
    y, sr = librosa.load(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    aggregated_chroma = np.mean(chroma, axis=1)  # Calculate the mean along the columns
    return aggregated_chroma

# Example dataset preparation
in_tune_files = ['wav/intune1.wav', 'wav/intune2.wav']
sharp_files = ['wav/sharp1.wav', 'wav/sharp2.wav']
flat_files = ['wav/flat1.wav', 'wav/flat2.wav']

in_tune_features = [extract_aggregated_features(file) for file in in_tune_files]
sharp_features = [extract_aggregated_features(file) for file in sharp_files]
flat_features = [extract_aggregated_features(file) for file in flat_files]

# Combining features and labels
X = np.vstack((in_tune_features, sharp_features, flat_features))

# Assigning labels to the features
n_samples_per_class = min(len(in_tune_features), len(sharp_features), len(flat_features))
y = np.hstack((np.ones(len(in_tune_features)), np.ones(len(sharp_features)) * 2, np.ones(len(flat_features)) * 3))

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
user_input_features = extract_aggregated_features('wav/a4/tuned_A4.wav')  # Extract features from user input
user_input_features_scaled = scaler.transform(user_input_features.reshape(1, -1))  # Scale user input features

# Predict individual notes
predictions = svm.predict(user_input_features_scaled)

# Analyze the distribution of predictions
in_tune_count = np.sum(predictions == 1)
sharp_count = np.sum(predictions == 2)
flat_count = np.sum(predictions == 3)

# Determine if the user is overall sharp or flat
if sharp_count > flat_count:
    print("Overall, the notes played are sharp.")
elif flat_count > sharp_count:
    print("Overall, the notes played are flat.")
else:
    print("The notes played are neither consistently sharp nor flat.")