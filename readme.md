# 🐱🐶 Transfer Learning - Classificação Gatos vs Cachorros

## 📋 Índice
- [Sobre o Projeto](#sobre-o-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Detalhamento das Funções](#detalhamento-das-funções)
- [Resultados Esperados](#resultados-esperados)
- [Troubleshooting](#troubleshooting)
- [Contribuição](#contribuição)

## 🎯 Sobre o Projeto

Este projeto implementa um classificador de imagens usando **Transfer Learning** para distinguir entre gatos e cachorros. Utilizamos o modelo VGG16 pré-treinado no ImageNet e aplicamos tanto **Feature Extraction** quanto **Fine-tuning** para obter alta performance com menos dados de treino.

### 🔧 Tecnologias Utilizadas
- **Python 3.8+**
- **TensorFlow/Keras** - Framework de Deep Learning
- **TensorFlow Datasets** - Dataset cats_vs_dogs
- **Matplotlib/Seaborn** - Visualizações
- **NumPy** - Operações numéricas
- **Scikit-learn** - Métricas de avaliação

## ⚙️ Pré-requisitos

### Sistema Operacional
- Windows 10/11, macOS, ou Linux
- Mínimo 8GB RAM (16GB recomendado)
- 5GB espaço livre em disco

### Python
- Python 3.8, 3.9, 3.10 ou 3.11
- **⚠️ Python 3.12+ não suportado pelo TensorFlow**
- Sistema 64-bit (obrigatório)

### Verificar Compatibilidade
```bash
# Verificar versão do Python
python --version

# Verificar se é 64-bit
python -c "import sys; print('64-bit' if sys.maxsize > 2**32 else '32-bit')"
```

## 🛠️ Instalação

### Opção 1: Google Colab (Recomendado) 🚀
1. Acesse: [Google Colab](https://colab.research.google.com)
2. Crie um novo notebook
3. Execute apenas:
   ```bash
   !pip install tensorflow-datasets seaborn
   ```
4. **Vantagens:**
   - ✅ TensorFlow já instalado
   - ✅ GPU gratuita
   - ✅ Zero configuração
   - ✅ 12GB RAM gratuita

### Opção 2: Ambiente Local

#### Instalação Básica
```bash
# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências
pip install tensorflow tensorflow-datasets matplotlib seaborn numpy scikit-learn
```

#### Usando Ambiente Virtual (Recomendado)
```bash
# Criar ambiente virtual
python -m venv venv_ml
cd venv_ml

# Ativar ambiente
# Windows:
Scripts\activate
# Linux/macOS:
source bin/activate

# Instalar dependências
pip install tensorflow tensorflow-datasets matplotlib seaborn numpy scikit-learn
```

#### Usando Conda
```bash
# Criar ambiente
conda create -n transfer_learning python=3.11
conda activate transfer_learning

# Instalar TensorFlow
conda install tensorflow

# Instalar outras dependências
pip install tensorflow-datasets seaborn
```

### Verificação da Instalação
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report

print("✅ Todas as bibliotecas instaladas com sucesso!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponível: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

## 🚀 Como Executar

### Passo 1: Download do Código
```bash
# Clone o repositório (se estiver no GitHub)
git clone [URL_DO_REPOSITORIO]
cd transfer-learning-cats-dogs

# Ou baixe o arquivo .py diretamente
```

### Passo 2: Executar o Projeto
```bash
# Se usando Python local
python transfer_learning_cats_dogs.py

# Se usando Jupyter/Colab
# Cole o código em células e execute sequencialmente
```

### Passo 3: Monitorar Execução
O projeto executará automaticamente:
1. ⬇️ Download do dataset (pode demorar na primeira vez)
2. 🔄 Pré-processamento dos dados
3. 🏗️ Criação do modelo com Transfer Learning
4. 🎯 Treinamento (Feature Extraction)
5. 🔧 Fine-tuning
6. 📊 Avaliação e visualização dos resultados

## 📁 Estrutura do Projeto

```
transfer-learning-cats-dogs/
│
├── 📄 transfer_learning_cats_dogs.py    # Código principal
├── 📄 README.md                         # Este arquivo
├── 📁 models/                          # Modelos salvos
│   ├── best_model.h5
│   ├── cats_dogs_transfer_learning_final.h5
│   └── cats_dogs_weights.h5
├── 📁 results/                         # Gráficos e resultados
└── 📁 data/                           # Dataset (baixado automaticamente)
```

## 🔍 Detalhamento das Funções

### 📊 Funções de Dados

#### `load_and_preprocess_data()`
**Propósito:** Carrega o dataset cats_vs_dogs usando TensorFlow Datasets

**O que faz:**
- Baixa automaticamente o dataset (25.000 imagens)
- Divide em 80% treino e 20% teste
- Retorna datasets estruturados

**Entrada:** Nenhuma
**Saída:** `ds_train`, `ds_test`, `info`

```python
# Exemplo de uso
train_ds, test_ds, dataset_info = load_and_preprocess_data()
```

#### `preprocess_image(image, label)`
**Propósito:** Pré-processa imagens para o modelo VGG16

**O que faz:**
- Redimensiona imagens para 224x224 pixels
- Converte para float32
- Normaliza valores de pixel

**Entrada:** 
- `image`: Tensor da imagem
- `label`: Label da classe (0=gato, 1=cachorro)

**Saída:** Imagem processada e label

### 🎨 Funções de Visualização

#### `plot_sample_images(dataset, num_images=12)`
**Propósito:** Visualiza amostras do dataset

**O que faz:**
- Plota 12 imagens em grid 3x4
- Mostra classe de cada imagem
- Ajusta automaticamente o layout

**Parâmetros:**
- `dataset`: Dataset do TensorFlow
- `num_images`: Quantidade de imagens (padrão: 12)

#### `plot_training_history(history_dict)`
**Propósito:** Plota gráficos do treinamento

**O que faz:**
- Gráfico de acurácia (treino vs validação)
- Gráfico de loss (treino vs validação)
- Identifica overfitting/underfitting

#### `plot_confusion_matrix(y_true, y_pred, class_names)`
**Propósito:** Gera matriz de confusão

**O que faz:**
- Calcula verdadeiros positivos/negativos
- Visualiza erros de classificação
- Retorna matriz para análise

### 🧠 Funções do Modelo

#### `create_transfer_learning_model()`
**Propósito:** Cria modelo usando Transfer Learning

**Arquitetura:**
```
Input (224x224x3)
↓
Pré-processamento VGG16
↓
VGG16 Base (congelado)
↓
GlobalAveragePooling2D
↓
Dropout(0.5)
↓
Dense(128, ReLU)
↓
Dropout(0.3)
↓
Dense(2, Softmax) → [Gato, Cachorro]
```

**O que faz:**
- Carrega VGG16 pré-treinado (sem top layers)
- Congela pesos do modelo base
- Adiciona camadas customizadas
- Retorna modelo compilado

#### `fine_tune_model(model, base_model, fine_tune_at=100)`
**Propósito:** Aplica fine-tuning ao modelo

**O que faz:**
- Descongela camadas superiores do VGG16
- Mantém camadas iniciais congeladas
- Reduz learning rate (10x menor)
- Permite ajuste fino dos filtros

**Parâmetros:**
- `fine_tune_at`: A partir de qual camada descongelar

### 🎯 Funções de Predição

#### `predict_single_image(model, dataset, index=0)`
**Propósito:** Testa modelo em imagem individual

**O que faz:**
- Pega uma imagem do dataset
- Faz predição com confiança
- Visualiza resultado comparando com verdade

#### `combine_histories(hist1, hist2)`
**Propósito:** Combina históricos de treinamento

**O que faz:**
- Une histórico de Feature Extraction + Fine-tuning
- Permite visualização contínua do treinamento

## 📈 Resultados Esperados

### 🎯 Performance
- **Acurácia no teste:** 85-95%
- **Tempo de treinamento:** 15-30 minutos (com GPU)
- **F1-Score:** > 0.90

### 📊 Saídas do Projeto
1. **Gráficos de Treinamento**
   - Evolução da acurácia
   - Evolução da loss
   - Identificação de overfitting

2. **Matriz de Confusão**
   - Erros de classificação
   - Precisão por classe

3. **Métricas Detalhadas**
   - Precision, Recall, F1-Score
   - Relatório de classificação

4. **Modelos Salvos**
   - `best_model.h5` - Melhor modelo durante treinamento
   - `cats_dogs_transfer_learning_final.h5` - Modelo final
   - `cats_dogs_weights.h5` - Apenas pesos

## 🔧 Configurações Ajustáveis

### Hiperparâmetros Principais
```python
# No início do código, você pode ajustar:
IMG_SIZE = 224          # Tamanho da imagem
BATCH_SIZE = 32         # Tamanho do batch
EPOCHS = 20             # Épocas de treinamento
LEARNING_RATE = 0.0001  # Taxa de aprendizado
```

### Personalização do Dataset
Para usar seu próprio dataset, substitua a função `load_and_preprocess_data()`:

```python
def load_custom_dataset():
    # Carregue suas imagens aqui
    # Organize em pastas: train/cats/, train/dogs/
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse'
    )
    
    return train_generator
```

## 🐛 Troubleshooting

### Problema: "Could not find TensorFlow"
**Solução:**
```bash
# Verificar Python
python --version  # Deve ser 3.8-3.11

# Verificar arquitetura
python -c "import platform; print(platform.architecture())"  # Deve ser 64-bit

# Reinstalar TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0
```

### Problema: "Out of Memory"
**Soluções:**
```python
# Reduzir batch size
BATCH_SIZE = 16  # ou 8

# Limitar uso de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

### Problema: Dataset não baixa
**Solução:**
```python
# Verificar conexão
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Ou baixar manualmente de:
# https://www.microsoft.com/en-us/download/details.aspx?id=54765
```

### Problema: Treinamento muito lento
**Soluções:**
1. **Use Google Colab** (GPU gratuita)
2. **Reduza epochs:** `EPOCHS = 10`
3. **Aumente batch size:** `BATCH_SIZE = 64`
4. **Skip fine-tuning:** Comente seção de fine-tuning

## 📚 Conceitos Importantes

### 🔄 Transfer Learning
**O que é:** Reutilização de conhecimento de um modelo pré-treinado

**Vantagens:**
- ✅ Menos dados necessários
- ✅ Treinamento mais rápido
- ✅ Melhor performance
- ✅ Menor custo computacional

### 🎯 Feature Extraction vs Fine-tuning

#### Feature Extraction
- Congela pesos do modelo pré-treinado
- Treina apenas camadas finais
- Mais rápido e estável

#### Fine-tuning
- Descongela algumas camadas do modelo base
- Ajusta filtros para domínio específico
- Melhor performance, mas requer cuidado

### 📊 Métricas de Avaliação

#### Acurácia
```
Acurácia = (TP + TN) / (TP + TN + FP + FN)
```

#### Precisão
```
Precisão = TP / (TP + FP)
```

#### Recall
```
Recall = TP / (TP + FN)
```

#### F1-Score
```
F1 = 2 × (Precisão × Recall) / (Precisão + Recall)
```

## 🎓 Próximos Passos

### Melhorias Sugeridas
1. **Data Augmentation**
   ```python
   train_datagen = ImageDataGenerator(
       rotation_range=20,
       width_shift_range=0.2,
       horizontal_flip=True
   )
   ```

2. **Outros Modelos Base**
   - ResNet50
   - InceptionV3
   - EfficientNet

3. **Ensemble Methods**
   - Combinar múltiplos modelos
   - Voting classifier

4. **Deploy**
   - API Flask/FastAPI
   - Interface web Streamlit
   - Aplicativo móvel

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/nova-funcionalidade`
3. Commit mudanças: `git commit -m 'Adiciona nova funcionalidade'`
4. Push: `git push origin feature/nova-funcionalidade`
5. Abra um Pull Request

### Áreas de Melhoria
- [ ] Implementar data augmentation
- [ ] Adicionar mais modelos base
- [ ] Criar interface web
- [ ] Otimizar hiperparâmetros
- [ ] Adicionar testes unitários

## 📝 Licença

Este projeto está sob licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Projeto Transfer Learning - DIO**
- 📧 Email: edsgom@gmail.com
- 🐙 GitHub: [@edsongom1](https://github.com/edsongom1)
- 💼 LinkedIn: [Edson Gomes](https://linkedin.com/in/edsongom1)

## 🙏 Agradecimentos

- **Digital Innovation One (DIO)** - Plataforma de ensino
- **TensorFlow Team** - Framework de Deep Learning
- **Karen Simonyan & Andrew Zisserman** - Criadores do VGG16
- **Kaggle** - Dataset cats vs dogs

---

## 📞 Suporte

Se encontrar problemas ou tiver dúvidas:

1. **Verifique a seção [Troubleshooting](#troubleshooting)**
2. **Abra uma Issue no GitHub**
3. **Use o Google Colab** para evitar problemas de configuração

**⚡ Dica:** O Google Colab é sempre a opção mais confiável para executar este projeto!

---

*Último update: Julho 2025*