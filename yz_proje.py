import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from google.colab import drive
import shutil

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

drive_source = Path('/content/drive/MyDrive/Colab Notebooks/brainds')
local_dest = Path('/content/temp_brainds')

if not os.path.exists(local_dest):
    print("Veriler hızlı diske alınıyor...")
    shutil.copytree(drive_source, local_dest)

train_dir = local_dest / "Train"
test_dir = local_dest / "Test"

def load_data_to_ram(directory):
    print(f"📂 {directory} klasöründeki resimler RAM'e yükleniyor (Lütfen Bekle)...")
    images = []
    labels = []

    for label_name in os.listdir(directory):
        class_dir = directory / label_name
        if not os.path.isdir(class_dir): continue

        print(f"  --> '{label_name}' sınıfı okunuyor...")
        for img_file in os.listdir(class_dir):
            try:
                img_path = str(class_dir / img_file)

                img = cv2.imread(img_path)
                if img is None: continue

                img = cv2.resize(img, (224, 224))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                images.append(img)
                labels.append(label_name)
            except Exception as e:
                pass

    return np.array(images), np.array(labels)

X_train_raw, y_train_raw = load_data_to_ram(train_dir)
X_test_raw, y_test_raw = load_data_to_ram(test_dir)

print("\n⚙️ Veriler işleniyor...")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_raw)
y_test_enc = le.transform(y_test_raw)

y_train_final = to_categorical(y_train_enc, num_classes=4)
y_test_final = to_categorical(y_test_enc, num_classes=4)

X_train_final = X_train_raw.astype('float32') / 255.0
X_test_final = X_test_raw.astype('float32') / 255.0

# Hafıza temizliği
del X_train_raw, y_train_raw, X_test_raw, y_test_raw
import gc
gc.collect()

print(f"\n✅ İŞLEM TAMAM! Tüm veri RAM'e yüklendi.")
print(f"Eğitim Verisi Şekli: {X_train_final.shape}")
print(f"Test Verisi Şekli: {X_test_final.shape}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("\n" + "="*40)
print(" BÖLÜM 2: KEŞİFSEL VERİ ANALİZİ (RAM)")
print("="*40)

class_names = list(le.classes_)
print(f"Tespit edilen sınıflar: {class_names}")

y_train_indices = np.argmax(y_train_final, axis=1)
y_train_names = [class_names[i] for i in y_train_indices]

plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_names, palette='viridis')
plt.title('Eğitim Verisinin Sınıf Dağılımı')
plt.xlabel('Tümör Türü')
plt.ylabel('Görüntü Sayısı')
plt.show()

print("\nRAM'deki veriden rastgele örnekler:")
fig, axes = plt.subplots(3, 4, figsize=(12, 10))

rand_indices = np.random.randint(0, len(X_train_final), 12)

for i, ax in enumerate(axes.flat):
    idx = rand_indices[i]
    img = X_train_final[idx]

    label_idx = np.argmax(y_train_final[idx])
    label_name = class_names[label_idx]

    ax.imshow(img)
    ax.set_title(label_name)
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f"Analiz Tamamlandı. Toplam Görüntü: {len(X_train_final)}")

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, DenseNet121,
    EfficientNetB0, InceptionV3, Xception,
    InceptionResNetV2, NASNetMobile
)

NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)

# ✅ SADECE MODEL İÇİNDE KULLANILACAK
data_augmentation_layers = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.01),
    layers.RandomTranslation(0.02, 0.02),
], name="gpu_augmentation")


def get_model_architecture(model_name, fine_tune_at=None):
    inputs = Input(shape=INPUT_SHAPE)

    x = data_augmentation_layers(inputs)

    # ======================================================

    if model_name == 'Selim_CNN':

        for filters in [32, 64, 128, 256]:
            x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)
            x = layers.MaxPooling2D()(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        return Model(inputs, outputs, name="Selim_CNN")


    if model_name == 'ResNet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == 'EfficientNetB0':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == 'InceptionResNetV2':
        base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == 'DenseNet121':
        base = DenseNet121(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == 'MobileNetV2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    if fine_tune_at is None:
        base.trainable = False   # 🔒 Feature extractor
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return Model(inputs, outputs, name=model_name)



models_to_train = [
    'Selim_CNN',
    'ResNet50',
    'EfficientNetB0',
    'InceptionResNetV2'
]

# Hiperparametre Araması
!pip install -q keras-tuner

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_tuning_model(hp):
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape)

    # Augmentation
    x = data_augmentation_layers(inputs)

    # Blok 1
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Parametre 1: Dropout Oranı (0.15 mi 0.25 mi?)
    hp_dropout1 = hp.Choice('dropout_1', values=[0.15, 0.25])
    x = layers.Dropout(hp_dropout1)(x)

    # Blok 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Blok 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)

    # Parametre 2: Dense Katmanındaki Nöron Sayısı (256 mı 512 mi?)
    hp_units = hp.Int('units', min_value=256, max_value=512, step=256)
    x = layers.Dense(hp_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    # Parametre 3: Learning Rate (Hız)
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    return model

# Tuner'ı Başlat (Hyperband Algoritması - Hızlıdır)
tuner = kt.Hyperband(
    build_tuning_model,
    objective='val_accuracy',
    max_epochs=5,     # Hızlı sonuç için az epoch
    factor=3,
    directory='my_dir',
    project_name='beyin_tumoru_optimizasyon'
)

# Aramayı Durduracak Erken Durdurma
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

print("🔍 Arama Başlıyor (Biraz sürebilir)...")

# Küçük bir veri parçasıyla arama yap (Tüm veriyle yaparsak günlerce sürer)
# Sadece ilk 500 resimle en iyi parametreyi bulalım, sonra asıl modelde kullanırız.
X_tune = X_train_final[:500]
y_tune = y_train_final[:500]
X_val_tune = X_train_final[500:600]
y_val_tune = y_train_final[500:600]

tuner.search(X_tune, y_tune, epochs=5, validation_data=(X_val_tune, y_val_tune), callbacks=[stop_early])

# En İyi Parametreleri Al
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n" + "="*50)
print("✅ OPTİMİZASYON TAMAMLANDI! EN İYİ DEĞERLER:")
print(f"1. En İyi Dropout Oranı: {best_hps.get('dropout_1')}")
print(f"2. En İyi Nöron Sayısı: {best_hps.get('units')}")
print(f"3. En İyi Learning Rate: {best_hps.get('learning_rate')}")
print("="*50)
print("Bu değerler raporda 'Parameter Optimization' başlığı altına eklenecektir.")

import tensorflow as tf
import gc
tf.keras.backend.clear_session()
gc.collect()

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import gc

# ===============================
# SABİT AYARLAR (L4 UYUMLU)
# ===============================
EPOCHS = 50
FOLDS = 2
BATCH_SIZE = 32          # L4 için ideal
INIT_LR = 3e-4           # DÜŞÜRÜLDÜ (kritik)
INPUT_SHAPE = (224,224,3)
NUM_CLASSES = 4

print("🚀 L4 GPU | STABİL TRAINING BAŞLADI")

# ===============================
# DATA AUGMENTATION (GPU)
# ===============================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1)
])

# ===============================
# MODEL (DÜZELTİLDİ)
# ===============================
def build_selim_cnn():
    inputs = Input(shape=INPUT_SHAPE)

    # 🔴 EN KRİTİK SATIR
    x = layers.Rescaling(1./255)(inputs)

    x = augmentation(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return models.Model(inputs, outputs, name="Selim_CNN")

# ===============================
# K-FOLD
# ===============================
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_no = 1
fold_scores = []

for train_idx, val_idx in kf.split(X_train_final):
    print(f"\n>>> Fold {fold_no}/{FOLDS}")

    X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
    y_tr, y_val = y_train_final[train_idx], y_train_final[val_idx]

    model = build_selim_cnn()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INIT_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.3, monitor='val_loss')
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    best_acc = max(history.history['val_accuracy'])
    fold_scores.append(best_acc)

    print(f"✅ Fold {fold_no} En İyi Val Acc: %{best_acc*100:.2f}")

    # ===============================
    # GRAFİK (DÜZGÜN & OKUNUR)
    # ===============================
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f"Fold {fold_no} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    fold_no += 1
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# ===============================
# ÖZET
# ===============================
print("\n🏁 TRAINING TAMAMLANDI")
for i, acc in enumerate(fold_scores, 1):
    print(f"Fold {i}: %{acc*100:.2f}")

print(f"📊 Ortalama Accuracy: %{np.mean(fold_scores)*100:.2f}")

import graphviz
from IPython.display import display, Image

# --- RENK PALETİ ---
COLOR_AUG       = "#E0E0E0"  # Gri (Augmentation)
COLOR_MODEL     = "#5DA5DA"  # Açık Mavi (Pre-trained Modeller)
COLOR_LAYER     = "#2E406E"  # Koyu Lacivert (Dense, Drop, GAP - Ara Katmanlar)
COLOR_CONCAT    = "#F28E2B"  # Turuncu (Birleştirme)
COLOR_TXT_W     = "#FFFFFF"  # Beyaz Yazı
COLOR_TXT_B     = "#000000"  # Siyah Yazı

# Sınıf Renkleri
CLASS_COLORS = {
    'Pituitary': '#4D4D4D',
    'No Tumor': '#999999',
    'Meningioma': '#999999',
    'Glioma': '#8C564B'
}

def draw_complete_hybrid_diagram():
    dot = graphviz.Digraph('Hybrid_Complete_Structure', format='png')
    dot.attr(rankdir='TB')    # Yukarıdan Aşağı
    dot.attr(splines='ortho') # Köşeli Çizgiler

    # Kutu Tasarımı
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='11', penwidth='0')

    # 1. GİRİŞ
    dot.node('Input', 'Input Image\n(224, 224, 3)', fillcolor='#FFFFFF', color='black', penwidth='1', fontcolor=COLOR_TXT_B)
    dot.node('Aug', 'Augmented\nImages', fillcolor=COLOR_AUG, fontcolor=COLOR_TXT_B)
    dot.edge('Input', 'Aug')

    # --- 2. SOL KOL (MobileNetV2 - DOLU YAPI) ---
    with dot.subgraph(name='cluster_left') as c:
        c.attr(style='invis')
        # Model
        c.node('MobNet', 'MobileNetV2', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)
        # GAP
        c.node('GAP1', 'GlobalAveragePooling2D', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        # Ara Katmanlar (Senin Değerlerinle Dolduruldu)
        c.node('L_Dense1', 'Dense\n256', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('L_Drop1', 'Dropout\n0.5', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('L_Dense2', 'Dense\n128', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('L_Drop2', 'Dropout\n0.5', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('L_DenseOut', 'Dense\n4', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)

        # Bağlantılar
        c.edge('MobNet', 'GAP1')
        c.edge('GAP1', 'L_Dense1')
        c.edge('L_Dense1', 'L_Drop1')
        c.edge('L_Drop1', 'L_Dense2')
        c.edge('L_Dense2', 'L_Drop2')
        c.edge('L_Drop2', 'L_DenseOut')

    # --- 3. SAĞ KOL (DenseNet121 - DOLU YAPI) ---
    with dot.subgraph(name='cluster_right') as c:
        c.attr(style='invis')
        # Model
        c.node('DenseNet', 'DenseNet121', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)
        # GAP
        c.node('GAP2', 'GlobalAveragePooling2D', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        # Ara Katmanlar (Senin Değerlerinle Dolduruldu)
        c.node('R_Dense1', 'Dense\n256', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('R_Drop1', 'Dropout\n0.5', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('R_Dense2', 'Dense\n128', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('R_Drop2', 'Dropout\n0.5', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)
        c.node('R_DenseOut', 'Dense\n4', fillcolor=COLOR_LAYER, fontcolor=COLOR_TXT_W)

        # Bağlantılar
        c.edge('DenseNet', 'GAP2')
        c.edge('GAP2', 'R_Dense1')
        c.edge('R_Dense1', 'R_Drop1')
        c.edge('R_Drop1', 'R_Dense2')
        c.edge('R_Dense2', 'R_Drop2')
        c.edge('R_Drop2', 'R_DenseOut')

    # Augmentation Bağlantıları
    dot.edge('Aug', 'MobNet')
    dot.edge('Aug', 'DenseNet')

    # --- 4. BİRLEŞTİRME (CONCATENATE) ---
    dot.node('Concat', 'Concatenate', fillcolor=COLOR_CONCAT, fontcolor=COLOR_TXT_W)
    dot.edge('L_DenseOut', 'Concat')
    dot.edge('R_DenseOut', 'Concat')

    # --- 5. FİNAL KUYRUK (SENİN DEĞERLERİN) ---
    # Sıra: Dense(256) -> Dropout(0.5) -> Dense(128) -> Dropout(0.5) -> Softmax
    # Referans resimdeki uzun kuyruğu senin değerlerinle yapıyoruz.

    dot.node('F_Dense1', 'Dense\n256', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W) # Resimde alt kısım Mavi
    dot.node('F_Drop1', 'Dropout\n0.5', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)
    dot.node('F_Dense2', 'Dense\n128', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)
    dot.node('F_Drop2', 'Dropout\n0.5', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)
    dot.node('Softmax', 'Dense 4\n(Softmax)', fillcolor=COLOR_MODEL, fontcolor=COLOR_TXT_W)

    dot.edge('Concat', 'F_Dense1')
    dot.edge('F_Dense1', 'F_Drop1')
    dot.edge('F_Drop1', 'F_Dense2')
    dot.edge('F_Dense2', 'F_Drop2')
    dot.edge('F_Drop2', 'Softmax')

    # --- 6. ÇIKIŞ SINIFLARI ---
    with dot.subgraph() as s:
        s.attr(rank='same')
        for class_name, color in CLASS_COLORS.items():
            s.node(class_name, class_name, fillcolor=color, fontcolor=COLOR_TXT_W, width='1.2')
            dot.edge('Softmax', class_name)

    return dot

print("🎨 Tam ve eksiksiz hibrit model şeması çiziliyor...")
try:
    diag = draw_complete_hybrid_diagram()
    display(diag)

    filename = 'Final_Complete_Hybrid_Diagram'
    diag.render(filename, format='png', cleanup=True)
    print(f"\n✅ Görsel Kaydedildi: {filename}.png")
    print("İndirmek için resme sağ tıklayıp 'Farklı Kaydet' diyebilirsin.")
except Exception as e:
    print(f"Hata: {e}")

import graphviz
from google.colab import files
from IPython.display import display, Image

# --- RENK PALETİ ---
COLOR_AUG       = "#E0E0E0"  # Gri (Augmentation)
COLOR_BLOCK     = "#E8F0FE"  # Çok Açık Mavi (Blok Arka Planı)
COLOR_CONV      = "#5DA5DA"  # Mavi (Conv Katmanları)
COLOR_BN_ACT    = "#4E79A7"  # Koyu Mavi (BN ve Aktivasyon)
COLOR_POOL      = "#F28E2B"  # Turuncu (Pooling - Boyut düşüren yer)
COLOR_DENSE     = "#2E406E"  # Lacivert (Dense/Classifier)
COLOR_TXT_W     = "#FFFFFF"
COLOR_TXT_B     = "#000000"

CLASS_COLORS = {
    'Pituitary': '#4D4D4D',
    'No Tumor': '#999999',
    'Meningioma': '#999999',
    'Glioma': '#8C564B'
}

def draw_selim_cnn():
    dot = graphviz.Digraph('Selim_CNN_Architecture', format='png')
    dot.attr(rankdir='TB')     # Yukarıdan Aşağı
    dot.attr(splines='ortho')  # Köşeli Çizgiler
    dot.attr(nodesep='0.6')    # Kutular arası boşluk

    # Kutu Tasarımı
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='10', penwidth='0')

    # 1. GİRİŞ
    dot.node('Input', 'Input Image\n(224, 224, 3)', fillcolor='#FFFFFF', color='black', penwidth='1', fontcolor=COLOR_TXT_B)
    dot.node('Aug', 'Augmented\nImages', fillcolor=COLOR_AUG, fontcolor=COLOR_TXT_B)
    dot.edge('Input', 'Aug')

    # --- BLOK 1 (32 Filtre) ---
    with dot.subgraph(name='cluster_block1') as c:
        c.attr(style='filled', color=COLOR_BLOCK, label='Block 1')
        c.node('Conv1', 'Conv2D\n(32, 3x3)', fillcolor=COLOR_CONV, fontcolor=COLOR_TXT_W)
        c.node('BN1', 'BatchNormalization', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Act1', 'LeakyReLU\n(alpha=0.1)', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Pool1', 'MaxPooling2D\n(2x2)', fillcolor=COLOR_POOL, fontcolor=COLOR_TXT_W)

        c.edge('Conv1', 'BN1')
        c.edge('BN1', 'Act1')
        c.edge('Act1', 'Pool1')

    # --- BLOK 2 (64 Filtre) ---
    with dot.subgraph(name='cluster_block2') as c:
        c.attr(style='filled', color=COLOR_BLOCK, label='Block 2')
        c.node('Conv2', 'Conv2D\n(64, 3x3)', fillcolor=COLOR_CONV, fontcolor=COLOR_TXT_W)
        c.node('BN2', 'BatchNormalization', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Act2', 'LeakyReLU\n(alpha=0.1)', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Pool2', 'MaxPooling2D\n(2x2)', fillcolor=COLOR_POOL, fontcolor=COLOR_TXT_W)

        c.edge('Conv2', 'BN2')
        c.edge('BN2', 'Act2')
        c.edge('Act2', 'Pool2')

    # --- BLOK 3 (128 Filtre) ---
    with dot.subgraph(name='cluster_block3') as c:
        c.attr(style='filled', color=COLOR_BLOCK, label='Block 3')
        c.node('Conv3', 'Conv2D\n(128, 3x3)', fillcolor=COLOR_CONV, fontcolor=COLOR_TXT_W)
        c.node('BN3', 'BatchNormalization', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Act3', 'LeakyReLU\n(alpha=0.1)', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Pool3', 'MaxPooling2D\n(2x2)', fillcolor=COLOR_POOL, fontcolor=COLOR_TXT_W)

        c.edge('Conv3', 'BN3')
        c.edge('BN3', 'Act3')
        c.edge('Act3', 'Pool3')

    # --- BLOK 4 (256 Filtre) ---
    with dot.subgraph(name='cluster_block4') as c:
        c.attr(style='filled', color=COLOR_BLOCK, label='Block 4')
        c.node('Conv4', 'Conv2D\n(256, 3x3)', fillcolor=COLOR_CONV, fontcolor=COLOR_TXT_W)
        c.node('BN4', 'BatchNormalization', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Act4', 'LeakyReLU\n(alpha=0.1)', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
        c.node('Pool4', 'MaxPooling2D\n(2x2)', fillcolor=COLOR_POOL, fontcolor=COLOR_TXT_W)

        c.edge('Conv4', 'BN4')
        c.edge('BN4', 'Act4')
        c.edge('Act4', 'Pool4')

    # BLOKLAR ARASI BAĞLANTILAR
    dot.edge('Aug', 'Conv1')
    dot.edge('Pool1', 'Conv2')
    dot.edge('Pool2', 'Conv3')
    dot.edge('Pool3', 'Conv4')

    # --- SINIFLANDIRMA (CLASSIFIER) ---
    # GAP -> Dense 256 -> BN -> Leaky -> Drop 0.4 -> Softmax

    dot.node('GAP', 'GlobalAveragePooling2D', fillcolor=COLOR_DENSE, fontcolor=COLOR_TXT_W)
    dot.edge('Pool4', 'GAP')

    dot.node('Dense256', 'Dense\n256', fillcolor=COLOR_DENSE, fontcolor=COLOR_TXT_W)
    dot.edge('GAP', 'Dense256')

    dot.node('BN_Final', 'BatchNormalization', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
    dot.edge('Dense256', 'BN_Final')

    dot.node('Act_Final', 'LeakyReLU\n(alpha=0.1)', fillcolor=COLOR_BN_ACT, fontcolor=COLOR_TXT_W)
    dot.edge('BN_Final', 'Act_Final')

    dot.node('Drop', 'Dropout\n0.4', fillcolor=COLOR_DENSE, fontcolor=COLOR_TXT_W)
    dot.edge('Act_Final', 'Drop')

    dot.node('Softmax', 'Dense 4\n(Softmax)', fillcolor=COLOR_CONV, fontcolor=COLOR_TXT_W)
    dot.edge('Drop', 'Softmax')

    # --- ÇIKIŞ SINIFLARI ---
    with dot.subgraph() as s:
        s.attr(rank='same')
        for class_name, color in CLASS_COLORS.items():
            s.node(class_name, class_name, fillcolor=color, fontcolor=COLOR_TXT_W, width='1.2')
            dot.edge('Softmax', class_name)

    return dot

# --- ÇİZDİR VE İNDİR ---
print("🎨 Selim_CNN mimarisi çiziliyor...")
try:
    diag = draw_selim_cnn()

    filename = 'Selim_CNN_Architecture'
    diag.render(filename, format='png', cleanup=True)

    display(Image(filename + '.png'))
    print("\n⬇️ Dosya indiriliyor...")
    files.download(filename + '.png')

except Exception as e:
    print(f"Hata: {e}")

import graphviz
from IPython.display import display, Image

def draw_exact_reference_flowchart():
    dot = graphviz.Digraph('Reference_Style_Methodology', format='png')
    dot.attr(rankdir='TB')     # Yukarıdan Aşağı Akış
    dot.attr(splines='ortho')  # Köşeli, dik açılı oklar

    # --- GENEL STİL AYARLARI (Referans Resimdeki Gibi Beyaz/Siyah) ---
    dot.attr('node', fontname='Arial', fontsize='11')

    # 1. GİRİŞ (Sensörler yerine MRI verisi)
    # Referans resimdeki en üstteki kutular
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Input1', 'Brain Tumor\nMRI Images', shape='rect', style='rounded', height='0.6')
        s.node('Input2', 'Labels\n(4 Classes)', shape='rect', style='rounded', height='0.6')

    # Birleşim noktası (Görünmez bir nokta)
    dot.node('JoinInputs', shape='point', width='0')
    dot.edge('Input1', 'JoinInputs', arrowhead='none')
    dot.edge('Input2', 'JoinInputs', arrowhead='none')

    # 2. İŞLEME ADIMLARI (Düz Kutu Zinciri)
    dot.node('Preproc', 'Image Preprocessing\n(Resize, Normalize)', shape='rect')
    dot.edge('JoinInputs', 'Preproc')

    dot.node('Aug', 'Data Augmentation', shape='rect')
    dot.edge('Preproc', 'Aug')

    dot.node('Feat', 'Feature Extraction\n(Implicit in DL Models)', shape='rect')
    dot.edge('Aug', 'Feat')

    # Veri Tabanı İkonu (Dataset)
    dot.node('Dataset', 'Prepared\nDataset', shape='cylinder', style='filled', fillcolor='white')
    dot.edge('Feat', 'Dataset')

    # 3. VERİ AYIRMA (SEPARATE DATASET) - Referans Resimdeki Kritik Kısım
    # Uzun bir ok ile sağa/aşağı geçiş
    dot.node('Separate', 'Separate Dataset\n(K-Fold / Split)', shape='rect')
    dot.edge('Dataset', 'Separate')

    # Training ve Test Data Silindirleri
    with dot.subgraph() as s:
        s.attr(rank='same')
        dot.node('TrainDB', 'Training Data', shape='cylinder', style='filled', fillcolor='white')
        dot.node('TestDB', 'Test Data', shape='cylinder', style='filled', fillcolor='white')

    # Ayrım Okları
    dot.edge('Separate', 'TrainDB')
    dot.edge('Separate', 'TestDB')

    # 4. MODELLER VE EĞİTİM (Gri Liste Kutusu)
    # Referans resimdeki "Machine and Deep Learning Methods" kutusu
    model_list_html = '''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">
      <TR><TD ALIGN="LEFT"><B>Deep Learning Models</B></TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">1. My Custom CNN</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">2. Hybrid (Mob+Dense)</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">3. DenseNet121</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">4. MobileNetV2</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">5. ResNet50</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">6. VGG16</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">7. EfficientNetB0</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">8. Xception</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">9. InceptionV3</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">10. IncResNetV2</TD></TR>
      <TR><TD ALIGN="LEFT" BGCOLOR="#E0E0E0">11. NASNetMobile</TD></TR>
    </TABLE>>'''

    # Gri Kutu
    dot.node('ModelList', model_list_html, shape='rect', style='filled', fillcolor='#F2F2F2')

    # "Build Model" Kutusu
    dot.node('Build', 'Build / Train\nModels', shape='rect')

    # Training Data -> Build Model
    dot.edge('TrainDB', 'Build')

    # Model Listesi Build Model'e beslenir (Ok yönü yukarıya doğru gibi görünür resimde ama mantıken oradadır)
    dot.edge('ModelList', 'Build', dir='back') # Model listesi buraya dahil

    # 5. DEĞERLENDİRME AKIŞI
    dot.node('Eval', 'Evaluate\nModel', shape='rect')

    # Test Data -> Evaluate (Sağ taraftan iniş)
    dot.edge('TestDB', 'Eval')

    # Build -> Evaluate (Sol taraftan geliş)
    dot.edge('Build', 'Eval')

    # 6. SONUÇLAR
    dot.node('Pred', 'Classification\nPrediction', shape='rect')
    dot.edge('Eval', 'Pred')

    dot.node('Perf', 'Performance\nEvaluation\n(Acc, Loss, ROC)', shape='rect')
    dot.edge('Pred', 'Perf')

    return dot

# --- ÇİZDİR VE İNDİR ---
print("🎨 Referans resimdeki yapıda (Training/Test Ayrımı Olan) şema çiziliyor...")
try:
    diag = draw_exact_reference_flowchart()

    filename = 'Ref_Style_Methodology_Split'
    diag.render(filename, format='png', cleanup=True)

    display(Image(filename + '.png'))

    from google.colab import files
    print("\n⬇️ Dosya indiriliyor...")
    files.download(filename + '.png')

except Exception as e:
    print(f"Hata: {e}")
