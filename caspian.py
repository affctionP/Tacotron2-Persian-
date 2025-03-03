import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
df=pd.read_csv("caspian7-10.csv")


label_column = 'bmi_category'

label_1_data = df[df[label_column] == 1]
label_0_data = df[df[label_column] == 0]
# # Randomly sample 700 rows from the label 1 data
downsampled_label_1_data = label_1_data.sample(n=1000, random_state=42)
downsampled_label_0_data = label_0_data.sample(n=1000, random_state=42)
# # # Combine the downsampled label 1 data with the rest of the data
rest_data = df[(df[label_column] != 1) & (df[label_column] != 0)]

downsampled_df = pd.concat([rest_data, downsampled_label_1_data,downsampled_label_0_data])

# # Shuffle the resulting DataFrame (optional)
downsampled_df = downsampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(downsampled_df[label_column].value_counts())
downsampled_df=downsampled_df.drop(columns=['weight', 'bmi','hip', 'wrist', 'waist','age','source',
                          'catage', 'grade', 'height','body_image','gender','systolic','diastoli'],axis=1)
# Assuming X is your input data and y are your labels
X = downsampled_df.iloc[:, :-1]# Example data
y = downsampled_df['bmi_category'] # Example labels

# Preprocessing
X = X / np.max(X)  # Normalize
y = to_categorical(y, num_classes=4)  # One-hot encoding

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
# Model Definition
model = Sequential()
model.add(Dense(96, input_shape=(112,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')