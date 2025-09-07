# import tensorflow as tf
# from tensorflow.keras import layers, models

# IMG_SIZE = 224

# def build_model():
#     base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),weights='imagenet')
#     base_model.trainable = False  # fine-tune later
    
#     inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
#     x = tf.keras.applications.efficientnet.preprocess_input(inputs)
#     x = base_model(x, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.3)(x)
#     outputs = layers.Dense(1, activation='sigmoid')(x)
    
#     model = models.Model(inputs, outputs)
#     return model
########################################################
# from typing import List
# import tensorflow as tf
# from tensorflow.keras import layers, models


# def build_model(num_classes: int, img_size: int = 320, dropout: float = 0.3) -> tf.keras.Model:
#     base = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(img_size, img_size, 3),weights="imagenet")
#     base.trainable = False # warmup; we will unfreeze top layers later


#     inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
#     x = tf.keras.applications.efficientnet.preprocess_input(inputs)


#     aug = tf.keras.Sequential([
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(0.05),
#         layers.RandomZoom(0.1),
#         layers.RandomContrast(0.1),
#     ], name="augment")


#     x = aug(x)
#     x = base(x, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(dropout)(x)


#     if num_classes == 2:
#         outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
#     else:
#         outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)


#     model = models.Model(inputs, outputs, name="eye_disease_classifier")
#     return model


import torch.nn as nn
import torch.nn.functional as F

class EyeDiseaseNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EyeDiseaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x