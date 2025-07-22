# Transfer Learning - Classificação de Gatos vs Cachorros
# Projeto desenvolvido para aplicar conceitos de Transfer Learning usando TensorFlow/Keras
# Edson G Chaves - 07-2025

# Instalação e importação das bibliotecas necessárias
!pip install tensorflow tensorflow-datasets matplotlib seaborn

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import zipfile
import requests
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================

# Configurações do projeto
IMG_SIZE = 224  # Tamanho padrão para VGG16
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 2
LEARNING_RATE = 0.0001

# Criação de diretórios
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

def load_and_preprocess_data():
    """
    Carrega e pré-processa o dataset cats_vs_dogs usando TensorFlow Datasets
    """
    print("Carregando dataset cats_vs_dogs...")
    
    # Carregamento do dataset
    (ds_train, ds_test), info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],  # 80% treino, 20% teste
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    
    print(f"Dataset info: {info}")
    print(f"Total de imagens de treino: {info.splits['train'].num_examples * 0.8:.0f}")
    print(f"Total de imagens de teste: {info.splits['train'].num_examples * 0.2:.0f}")
    
    return ds_train, ds_test, info

def preprocess_image(image, label):
    """
    Pré-processa as imagens para o modelo
    """
    # Redimensiona a imagem
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalização para VGG16 (valores entre 0-255 para preprocess_input)
    image = tf.cast(image, tf.float32)
    
    return image, label

# Carregamento dos dados
train_ds, test_ds, dataset_info = load_and_preprocess_data()

# Pré-processamento
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Separação de validação
val_size = int(0.2 * tf.data.experimental.cardinality(train_ds).numpy())
val_ds = train_ds.take(val_size)
train_ds = train_ds.skip(val_size)

# Otimização de performance
train_ds = train_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Batches de treino: {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Batches de validação: {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"Batches de teste: {tf.data.experimental.cardinality(test_ds).numpy()}")

# =============================================================================
# VISUALIZAÇÃO DOS DADOS
# =============================================================================

def plot_sample_images(dataset, num_images=9):
    """
    Plota algumas imagens de exemplo do dataset
    """
    plt.figure(figsize=(12, 12))
    class_names = ['Gato', 'Cachorro']
    
    for i, (image, label) in enumerate(dataset.unbatch().take(num_images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype('uint8'))
        plt.title(f'Classe: {class_names[label.numpy()]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("Visualizando amostras do dataset:")
plot_sample_images(train_ds)

# =============================================================================
# CRIAÇÃO DO MODELO COM TRANSFER LEARNING
# =============================================================================

def create_transfer_learning_model():
    """
    Cria modelo usando Transfer Learning com VGG16
    """
    # Carrega o modelo VGG16 pré-treinado (sem as camadas densas finais)
    base_model = VGG16(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,  # Remove as camadas densas finais
        weights='imagenet'  # Usa pesos pré-treinados do ImageNet
    )
    
    # Congela as camadas do modelo base
    base_model.trainable = False
    
    # Cria o modelo completo
    model = models.Sequential([
        # Pré-processamento específico do VGG16
        layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(x)),
        
        # Modelo base
        base_model,
        
        # Camadas personalizadas para classificação
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model, base_model

# Criação do modelo
model, base_model = create_transfer_learning_model()

# Compilação do modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Resumo do modelo
print("Arquitetura do modelo:")
model.summary()

print(f"\nNúmero de camadas do modelo base (VGG16): {len(base_model.layers)}")
print(f"Número de camadas treináveis: {len([l for l in model.layers if l.trainable])}")

# =============================================================================
# CALLBACKS PARA TREINAMENTO
# =============================================================================

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint]

# =============================================================================
# TREINAMENTO INICIAL (FEATURE EXTRACTION)
# =============================================================================

print("Iniciando treinamento (Feature Extraction)...")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# FINE-TUNING (OPCIONAL)
# =============================================================================

def fine_tune_model(model, base_model, fine_tune_at=100):
    """
    Aplica fine-tuning no modelo
    """
    # Descongela o modelo base
    base_model.trainable = True
    
    # Congela as camadas iniciais
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompila com learning rate menor
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Camadas treináveis após fine-tuning: {len([l for l in model.layers if l.trainable])}")
    
    return model

# Aplicar fine-tuning
print("\nAplicando Fine-tuning...")
model = fine_tune_model(model, base_model, fine_tune_at=15)

# Treinamento com fine-tuning
history_fine = model.fit(
    train_ds,
    epochs=10,  # Menos épocas para fine-tuning
    initial_epoch=len(history.history['loss']),
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

# Combinar históricos
def combine_histories(hist1, hist2):
    """Combina dois históricos de treinamento"""
    combined = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined

full_history = combine_histories(history, history_fine)

# =============================================================================
# VISUALIZAÇÃO DOS RESULTADOS DE TREINAMENTO
# =============================================================================

def plot_training_history(history_dict):
    """
    Plota gráficos de loss e accuracy durante o treinamento
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot da acurácia
    axes[0].plot(history_dict['accuracy'], label='Treino')
    axes[0].plot(history_dict['val_accuracy'], label='Validação')
    axes[0].set_title('Acurácia do Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Acurácia')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot da loss
    axes[1].plot(history_dict['loss'], label='Treino')
    axes[1].plot(history_dict['val_loss'], label='Validação')
    axes[1].set_title('Loss do Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

print("Visualizando histórico de treinamento:")
plot_training_history(full_history)

# =============================================================================
# AVALIAÇÃO DO MODELO
# =============================================================================

print("Avaliando modelo no conjunto de teste...")

# Avaliação no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")
print(f"Loss no conjunto de teste: {test_loss:.4f}")

# Predições para matriz de confusão
y_pred = []
y_true = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Matriz de confusão
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plota matriz de confusão
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()
    
    return cm

class_names = ['Gato', 'Cachorro']
cm = plot_confusion_matrix(y_true, y_pred, class_names)

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =============================================================================
# TESTE COM IMAGENS INDIVIDUAIS
# =============================================================================

def predict_single_image(model, dataset, index=0):
    """
    Faz predição em uma única imagem
    """
    class_names = ['Gato', 'Cachorro']
    
    # Pega uma imagem do dataset de teste
    for i, (image, true_label) in enumerate(dataset.unbatch().take(index + 1)):
        if i == index:
            # Prepara a imagem para predição
            img_array = tf.expand_dims(image, 0)
            
            # Faz a predição
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Visualiza resultado
            plt.figure(figsize=(8, 6))
            plt.imshow(image.numpy().astype('uint8'))
            plt.title(f'Verdadeiro: {class_names[true_label.numpy()]}\n'
                     f'Predito: {class_names[predicted_class]} '
                     f'(Confiança: {confidence:.2f})')
            plt.axis('off')
            plt.show()
            
            break

print("Testando predições em imagens individuais:")
for i in range(3):
    predict_single_image(model, test_ds, index=i)

# =============================================================================
# SALVAMENTO DO MODELO FINAL
# =============================================================================

# Salva o modelo final
model.save('models/cats_dogs_transfer_learning_final.h5')
print("Modelo salvo em 'models/cats_dogs_transfer_learning_final.h5'")

# Salva apenas os pesos
model.save_weights('models/cats_dogs_weights.h5')
print("Pesos salvos em 'models/cats_dogs_weights.h5'")

# =============================================================================
# RESUMO DOS RESULTADOS
# =============================================================================

print("\n" + "="*60)
print("RESUMO DOS RESULTADOS")
print("="*60)
print(f"Acurácia final no teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Loss final no teste: {test_loss:.4f}")
print(f"Épocas de treinamento: {len(full_history['loss'])}")
print(f"Melhor acurácia de validação: {max(full_history['val_accuracy']):.4f}")

# Estatísticas da matriz de confusão
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"\nMétricas para classe 'Cachorro':")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

print("\n" + "="*60)
print("PROJETO CONCLUÍDO COM SUCESSO!")
print("="*60)

# =============================================================================
# INSTRUÇÕES PARA USO
# =============================================================================

print("""
INSTRUÇÕES PARA USAR ESTE CÓDIGO:

1. Execute este código no Google Colab
2. O dataset será baixado automaticamente via TensorFlow Datasets
3. O modelo será treinado usando Transfer Learning com VGG16
4. Os resultados serão salvos na pasta 'models/'

PERSONALIZAÇÃO:
- Para usar seu próprio dataset, substitua a função load_and_preprocess_data()
- Ajuste as configurações no início do código (IMG_SIZE, BATCH_SIZE, etc.)
- Experimente diferentes modelos base (ResNet50, InceptionV3, etc.)

PRÓXIMOS PASSOS:
- Fazer upload do projeto para o GitHub da DIO
- Experimentar com data augmentation
- Testar outros modelos pré-treinados
- Criar uma interface web para classificação
""")
