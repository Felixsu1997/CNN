#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[2]:


data_dir = r'C:\Users\ASUS\Desktop\新增資料夾'  # path
img_height, img_width = 640, 640
batch_size = 8
num_classes = 3  # type class num 


# In[3]:


atagen = keras.preprocessing.image.ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training')

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation')


# In[ ]:


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # output
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator


# In[ ]:


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


def classify_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    class_name = list(train_generator.class_indices.keys())[class_index]
    confidence = 100 * np.max(prediction)
    return class_name, confidence


# In[ ]:


test_dir = r'C:\Users\ASUS\Desktop\新增資料夾'  # path

image_paths = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir) if filename.endswith(('.jpg', '.jpeg', '.png','.JPG'))]
print(f"找到 {len(image_paths)} ")

classification_results = []

for img_path in image_paths:
    class_name, confidence = classify_image(img_path)
    base_name = f"{class_name}_{os.path.basename(img_path)}"
    new_path = os.path.join(test_dir, base_name)
    
    counter = 1
    while os.path.exists(new_path):
        name, ext = os.path.splitext(base_name)
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(test_dir, new_name)
        counter += 1
    
    os.rename(img_path, new_path)
    classification_results.append({'Image': os.path.basename(new_path), 'Class': class_name})
    print(f"圖像 {os.path.basename(img_path)} 分類{class_name}，信心度 {confidence:.2f}%")

df = pd.DataFrame(classification_results)

print(df.head())  

output_csv_path = 'classification_results.csv'
df.to_csv(output_csv_path, index=False)
print(f"done '{output_csv_path}' file")


# In[ ]:


true_labels = validation_generator.classes
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=list(train_generator.class_indices.keys()))

cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

