import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle

# ✅ Xây dựng mô hình CNN nâng cao
def build_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# ✅ Đường dẫn tới thư mục chứa dữ liệu
train_dir = 'backend/cifar-10-upscaled'

# ✅ Chuẩn bị dữ liệu với Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2
)

# ✅ Tạo generator cho training & validation
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# ✅ Tạo tập test (20% từ dữ liệu validation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

test_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# ✅ Xây dựng mô hình CNN
input_shape = (32, 32, 3)
num_classes = 10
model = build_cnn(input_shape, num_classes)

# ✅ Biên dịch mô hình với Adam Optimizer
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Callbacks: Giảm Learning Rate và Early Stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ✅ Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=80,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping]
)

# ✅ Lưu history vào file .pkl
with open('backend/models/history_cnn2.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# ✅ Lưu mô hình dưới dạng .h5
model.save('backend/models/cnn_model2.h5')

# ✅ Đánh giá mô hình trên tập validation
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# ✅ Đánh giá mô hình trên tập test & LƯU KẾT QUẢ
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# ✅ Lưu kết quả test vào file .pkl
test_results = {"test_loss": test_loss, "test_accuracy": test_accuracy}
with open('backend/models/test_results.pkl', 'wb') as f:
    pickle.dump(test_results, f)
