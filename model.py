import os
import joblib
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def _load_audio_files(audio_dir, sr=16000):
    audio_data = []
    labels = []

    for sub_dir in os.listdir(audio_dir):
        for sub_sub_dir in os.listdir(os.path.join(audio_dir, sub_dir)):
            temp_labels = dict()

            for file in os.listdir(os.path.join(audio_dir, sub_dir, sub_sub_dir)):
                if file.endswith('.trans.txt'):
                    trans_file = open(os.path.join(audio_dir, sub_dir, sub_sub_dir, file), 'r')
                    for line in trans_file.readlines():
                        temp_labels[line.split(' ')[0]] = line.split(' ', 1)[1].strip()
                    break

            for file in os.listdir(os.path.join(audio_dir, sub_dir, sub_sub_dir)):
                if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.flac'):
                    file_path = os.path.join(audio_dir, sub_dir, sub_sub_dir, file)
                    label = temp_labels[file.split('.')[0]]
                    data, _ = librosa.load(file_path, sr=sr)
                    audio_data.append(data)
                    labels.append(label)

            temp_labels.clear()

    return audio_data, labels

def _extract_mfcc_features(audio_data, sr=16000, n_mfcc=13):
    mfcc_features = []
    
    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)  # Mittelwert über die Zeitachse
        mfcc_features.append(mfcc)
    
    return np.array(mfcc_features)

def _normalize_features(features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features, scaler
    else:
        return scaler.transform(features)

def _save_prepared_data(features, labels, scaler, file_name='prepared_data.npz'):
    np.savez(file_name, features=features, labels=labels)
    joblib.dump(scaler, 'scaler.pkl')

def setup(audio_dir):
    audio_data, labels = _load_audio_files(audio_dir)
    print('Audio-Daten wurden geladen.')
    
    mfcc_features = _extract_mfcc_features(audio_data)
    print('MFCC-Merkmale wurden extrahiert.')
    
    normalized_features, scaler = _normalize_features(mfcc_features)
    print('Merkmale wurden normalisiert.')
    
    _save_prepared_data(normalized_features, labels, scaler)
    print('Vorbereitete Daten wurden gespeichert.')

def train():
    data = np.load('prepared_data.npz')
    features = data['features']
    labels = data['labels']
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(150,100,50), activation='relu', solver='sgd', max_iter=1200, random_state=42)
    
    model.fit(X_train, y_train)
    joblib.dump(model, 'speech_command_model.pkl')
    
    test_acc = model.score(X_test, y_test)
    print(f"Testgenauigkeit: {test_acc}")

def run(audio_file, threshold: float = 0.45):
    if threshold < 0 or threshold > 1:
        raise ValueError("Der Schwellenwert muss zwischen 0 und 1 liegen.")

    model = joblib.load('speech_command_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    data, sr = librosa.load(audio_file, sr=16000)
    
    mfcc_features = _extract_mfcc_features([data])
    normalized_features = _normalize_features(mfcc_features, scaler)
    
    probabilities = model.predict_proba(normalized_features)
    prediction = model.predict(normalized_features)
    
    print(f"Wahrscheinlichkeiten für jede Klasse: {probabilities[0]}")

    max_probability = np.max(probabilities)

    if prediction == "unknown" or max_probability < threshold:
        print("Unklare Vorhersage. Keine Aktion wird ausgeführt.")
    else:
        print(f"Vorhersage: {prediction} mit einer Wahrscheinlichkeit von {round(max_probability * 100, 2)}%")

if __name__ == '__main__':
    run('data/test/licht-0014.wav')
