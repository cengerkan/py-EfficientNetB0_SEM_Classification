# 📌 SEM Nano-Material Classification with EfficientNetB0  
### 🔬 Scanning Electron Microscope (SEM) Image Classification  

Bu proje, **EfficientNetB0** tabanlı bir derin öğrenme modeli kullanarak SEM görüntülerinden **Fibres, Nanowires, Particles ve Powder** olmak üzere dört nano-malzeme tipini otomatik olarak sınıflandırır.  
Model iki aşamalı olarak eğitilir:  
- **Phase 1 – Transfer Learning (Base model frozen)**  
- **Phase 2 – Fine-Tuning (Base modelin son katmanları açılır)**  

Sonuçlar arasında:  
✔ Confusion Matrix  
✔ Eğitim Grafik Çıktıları  
✔ Tek Görüntü Tahmin Fonksiyonu  
✔ Modele ait H5 dosyası  
✔ Model bilgilerini içeren JSON dosyası  

---

## 📁 Proje Yapısı
project/
│
├── data100/ # Eğitim ve validation görüntüleri
│ ├── Fibres/
│ ├── Nanowires/
│ ├── Particles/
│ └── Powder/

---

## ⚙️ Kullanılan Teknolojiler
- TensorFlow / Keras  
- EfficientNetB0  
- NumPy  
- Matplotlib  
- Scikit-Learn  
- Seaborn  

---

## 🚀 Model Eğitimi Nasıl Çalışıyor?

### **1️⃣ Veri Hazırlığı**
- Görseller 224×224 formatına getirilir.  
- `%80` train – `%20` validation split uygulanır.  
- Train için augmentation uygulanır: rotate, zoom, flip, shift.

---

### **2️⃣ Model Oluşturma (EfficientNetB0 + Custom Head)**

Model üstüne eklenen katmanlar:

- `GlobalAveragePooling2D`
- `BatchNormalization`
- `Dropout`
- `Dense(256, relu)`
- `BatchNormalization`
- `Dropout`
- `Dense(NUM_CLASSES, softmax)`

---

### **3️⃣ Phase 1 – Transfer Learning**
Base model dondurulur, sadece eklenen katmanlar eğitilir.  
- Epoch: 20  
- LR: 0.001  

---

### **4️⃣ Phase 2 – Fine-Tuning**
Base modelin son katmanları açılır.  
- İlk 100 katman dondurulur  
- Öğrenme oranı 10× düşürülür  
- Epoch: 30  

---

### **5️⃣ Sonuçlar**
- Validation Accuracy  
- Validation Loss  
- Top-2 Accuracy  
- Classification Report  
- Confusion Matrix  

---

## 📊 Üretilen Dosyalar

| Dosya | Açıklama |
|-------|----------|
| **efficientnet_b0_material_classifier.h5** | Final eğitimli model |
| **best_efficientnet_model.h5** | Validation accuracy'e göre en iyi epoch |
| **confusion_matrix.png** | Çıktı karışıklık matrisi |
| **training_history.png** | Accuracy ve Loss grafiklerini içerir |
| **model_info.json** | Model meta bilgileri |

---

## 🔍 Tek Görüntü Tahmin Örneği

```python
from tensorflow import keras
from predict import predict_single_image

model = keras.models.load_model("efficientnet_b0_material_classifier.h5")
class_labels = ["Fibres", "Nanowires", "Particles", "Powder"]

predict_single_image("test_image.jpg", model, class_labels)

---

## 🔍 Modeli Yüklemek için

from tensorflow import keras
model = keras.models.load_model("efficientnet_b0_material_classifier.h5")

