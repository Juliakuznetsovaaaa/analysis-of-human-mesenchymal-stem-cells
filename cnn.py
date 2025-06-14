
import os
import shutil

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt  # Добавляем matplotlib
from tensorflow.keras.preprocessing import image #Для загрузки и отображения отдельных картинок

# 1. Параметры
DATA_DIR = 'dir'  # Папка с двумя подпапками: good и bad
TEST_SIZE = 0.2  # Доля тестовых данных
IMG_SIZE = (299, 299)  # Размер для InceptionV3
BATCH_SIZE = 32
EPOCHS = 100



# 2. Автоматическое разделение на train/test
def split_data(src_dir, test_size=0.2):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # Создаем структуру папок
    for split in [train_dir, test_dir]:
        for class_name in ['good', 'bad']:
            os.makedirs(os.path.join(split, class_name), exist_ok=True)

    # Копируем файлы с рандомным разделением
    for class_name in ['good', 'bad']:
        src_class_dir = os.path.join(src_dir, class_name)
        files = [f for f in os.listdir(src_class_dir) if f.endswith(('.jpg', '.png', '.tiff', '.tif'))]
        np.random.shuffle(files)

        split_idx = int(len(files) * (1 - test_size))

        # Копируем в train
        for f in files[:split_idx]:
            shutil.copy(
                os.path.join(src_class_dir, f),
                os.path.join(train_dir, class_name, f)
            )

        # Копируем в test
        for f in files[split_idx:]:
            shutil.copy(
                os.path.join(src_class_dir, f),
                os.path.join(test_dir, class_name, f)
            )

    return train_dir, test_dir


# Вызываем функцию разделения
train_dir, test_dir = split_data(DATA_DIR, TEST_SIZE)

# 3. Создание генераторов данных
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# 4. Создание и обучение модели
base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Замораживаем базовые слои
for layer in base_model.layers:
    layer.trainable = False

# Добавляем новые слои
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)
class_names = list(train_generator.class_indices.keys())

# 5. Оценка и сохранение
loss, accuracy = model.evaluate(test_generator)
print(f'\nTest accuracy: {accuracy * 100:.2f}%')
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
model.save('classification_model.h5')

# 6. Вывод графиков и изображений
# График обучения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['Тренировка', 'Тест'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь модели')
plt.ylabel('Функция потерь')
plt.xlabel('Эпоха')
plt.legend(['Тренировка', 'Тест'], loc='upper left')

plt.tight_layout()
plt.show()

# Вывод нескольких изображений с предсказаниями
num_images = 0  # Количество изображений для отображения
test_files = test_generator.filepaths
test_labels = test_generator.labels

plt.figure(figsize=(15, 5 * num_images))
for i in range(num_images):
    # Случайное изображение из тестового набора
    idx = np.random.randint(0, len(test_files))
    img_path = test_files[idx]
    true_label = class_names[test_labels[idx]]

    # Загрузка и предварительная обработка изображения
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # rescale


    # Предсказание
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Отображение изображения и информации
    plt.subplot(num_images, 1, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}, Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()


# Добавляем Grad-CAM визуализацию
def grad_cam(model, img_array, layer_name='conv2d_93'):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Вывод изображений с Grad-CAM
i=0
plt.figure(figsize=(20, 8))

while i < num_images:
    # Создаем новую фигуру каждые 5 изображений
    if i % 5 == 0:
        plt.figure(figsize=(20, 8))

    # Выбор случайного изображения
    idx = np.random.randint(0, len(test_files))
    img_path = test_files[idx]
    true_label = class_names[test_labels[idx]]

    # Загрузка и предобработка
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.

    # Предсказание
    prediction = model.predict(img_array_expanded, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Grad-CAM
    heatmap = grad_cam(model, img_array_expanded)
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Отрисовка
    plt.subplot(2, 5, (i % 5) + 1)  # Верхний ряд (1-5)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {predicted_class} ({confidence:.2f})")
    plt.axis('off')

    plt.subplot(2, 5, (i % 5) + 6)  # Нижний ряд (6-10)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis('off')

    # Показываем график после каждых 5 изображений
    if (i + 1) % 5 == 0:
        plt.tight_layout()
        plt.show()

    i += 1