# 🚴‍♂️ Bisiklet Paylaşım Talep Tahmini Projesi

Bu proje, bisiklet paylaşım sistemlerinde kullanıcı talebini tahmin etmek için makine öğrenimi tekniklerini kullanmaktadır. Amaç, kullanıcıların bisikletleri hangi zaman dilimlerinde daha fazla talep ettiğini belirlemektir.

⚠️ Not
3D grafiklerim ve görselleştirmelerim maalesef gözükmüyor. Bu durum, bazı tarayıcı veya platform uyumsuzluklarından kaynaklanabilir.

## 🔗 Kaggle Veri Seti
[Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)

## 🔗 Hugging Face Uygulaması
[Bisiklet - Hugging Face Space](https://huggingface.co/spaces/btulftma/bikepredict)

## 📊 Proje Aşamaları
1. **Veri Yükleme**:
   - Eğitim (`train.csv`) ve test (`test.csv`) veri setleri yüklenir.

2. **Veri Ön İşleme**:
   - Tarih ve saat bilgileri işlenir; saat, gün, ay, yıl ve haftanın günü gibi yeni özellikler eklenir.
   - Bisiklet kiralama sayısı logaritmik dönüşüme tabi tutulur.

3. **Keşifsel Veri Analizi (EDA)**:
   - Bisiklet kiralama sayısının dağılımı, saatlik ve mevsimsel trendler görselleştirilir.
   - Hava durumu, sıcaklık, nem gibi değişkenlerin kiralama sayısına etkisi incelenir.

4. **Özellik Mühendisliği**:
   - Kategorik değişkenler için one-hot encoding uygulanır ve orantısal özellikler oluşturulur.
   - Aykırı değerler temizlenir.

5. **Modelleme**:
   - LightGBM, Random Forest ve Gradient Boosting gibi modeller ile tahmin yapılır.
   - GridSearchCV ile model hiperparametre optimizasyonu gerçekleştirilir.

6. **Model Değerlendirme**:
   - Modellerin performansı RMSLE (Root Mean Squared Logarithmic Error) ile değerlendirilir.

7. **Tahminler ve Sonuçlar**:
   - En iyi model ile test veri seti üzerinde tahmin yapılır ve sonuçlar bir CSV dosyası olarak kaydedilir.

## 📈 Model Performansı
- Kaggle skoru: `0.38706`
- En iyi modelin RMSLE değeri belirtilmiştir.

## 🛠️ Kullanılan Kütüphaneler
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `xgboost`, `lightgbm`, `sklearn` (makine öğrenimi ve modelleme için)

## 📊 EDA Kısmı
Keşifsel Veri Analizi (EDA) sürecinde:
- Bisiklet kiralama sayısının dağılımı ve aylık/haftalık trendler görselleştirilmiştir.
- Korelasyon analizi yapılarak değişkenler arasındaki ilişkiler incelenmiştir.

Bu proje, bisiklet paylaşım sistemlerinde kullanıcı talebini daha iyi anlamak ve yönetmek için önemli içgörüler sağlamaktadır. Elde edilen model, bisikletlerin hangi zaman dilimlerinde daha fazla talep göreceğini tahmin etmeye yardımcı olmaktadır.
