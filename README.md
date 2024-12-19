# Aygaz Görüntü İşleme Bootcamp 🎯
## Authors
* **İlayda Uçan ** - [Slippers3310](https://github.com/Slippers3310)

## Lisans

Bu proje, MIT Lisansı altında lisanslanmıştır. Lisans detayları için [LICENSE](LICENSE) dosyasına bakabilirsiniz.

# Proje Adı
Aygaz Görüntü İşleme Bootcamp
📌 Projenin Amacı :
Bu proje, hayvan sınıflandırması problemini çözmek için bir Convolutional Neural Network (CNN) modeli geliştirmeyi hedeflemektedir.
**Görsel veri analizi ve işleme yeteneği yüksek olan bu model, farklı hayvan türlerini (örneğin, "fil", "tilki", "tavşan") tanımlayarak sınıflandırma yapar. **

Amacımız görüntü işlemeve derin öğrenme tekniklerini kullanarak :
🖼️ Görsel verilerden anlamlı özellikler çıkarmak,
🤖 Modeli genelleme yeteneği yüksek bir sınıflandırıcı olarak eğitmek,
🌍 Gerçek dünyadaki veri manipülasyonu ve varyasyonlarına dayanıklı bir model geliştirmektir.


# 📚 **1- Gerekli Kütüphanelerin Yüklenmesi** 
###    👉 os ve shutil: Dosya ve klasör yönetimi işlemlerini kolaylaştırmak için.
###    👉 cv2 (OpenCV): Görüntü işleme, boyutlandırma ve analiz için.
###    👉 numpy: Sayısal hesaplamalar, dizi işlemleri ve veri manipülasyonu için.
###    👉 ImageDataGenerator: Görüntü veri artırma (data augmentation) ve ön işleme için.
###    👉 train_test_split: Veriyi eğitim ve test setlerine ayırmak için.
###    👉 Model, Dense, Flatten, Dropout, BatchNormalization: Derin öğrenme modellerini oluşturmak ve katman eklemek için Keras araçları.
###    👉 matplotlib: Grafikler ve görsellerle sonuçları analiz etmek ve görselleştirmek için.
###    👉 keras: Derin öğrenme modellerini oluşturmak ve eğitmek için genel işlevsellik sağlar.
###    👉 models, layers: Keras ile model mimarileri ve özel katmanlar oluşturmak için.


# 🧑‍💻 2- Veri Setinin Hazırlanması :
## 📁 Veri Hazırlığı:
    
### 🎯 Seçili Sınıflar: "Fil", "Tavşan", "Tilki" gibi belirli hayvan sınıflarını seçtik.
### ⚖️ Veri Dengeleme: Her sınıftan eşit sayıda görüntü alınarak dengesizlik giderildi.

Bu bölümde, modeliniz için verilerin uygun şekilde hazırlanması, filtrelenmesi ve eğitim ile test setlerine ayrılması işlemleri yapılır. Her adımı detaylıca inceleyelim:

# *🧹 2.1 Veri Setinin Filtrelenmesi*

**Amaç: Veri setinden yalnızca belirli sınıflara ait görüntüler alınarak bu sınıflar için verinin dengelenmesi sağlanır.**

### **🌟 Önemli Adımlar:**
##### Klasör Yolları: source_dir ve target_dir, verilerin bulunduğu klasörün yolu ve filtrelenmiş verilerin kaydedileceği yeni klasörün yolu.
##### Sınıflar Seçimi: selected_classes ile kullanılacak 10 farklı sınıf seçilir. Bu, modelin sadece bu sınıflara odaklanmasını sağlar.
##### Görüntü Sayısı Limiti: images_per_class = 650 ile her sınıf için maksimum 650 resim seçilir, böylece veri seti dengelenir.
##### Veri Filtreleme: Her sınıf için, klasördeki resimler belirli bir sayıya kadar alınır (i >= images_per_class ile sınırlandırılır), ardından bu resimler target_dir altındaki uygun alt klasöre kopyalanır.

   ### 📝 Sonuç: "Veri seti hazırlandı ve dengelendi." mesajı, filtreleme işleminin başarıyla tamamlandığını gösteriyor.# 📏 2.2 - Boyutlandırma ve Normalizasyon
 **Tüm görseller aynı boyutta ve ölçeklendirilmiş şekilde işlendi.**

### -> load_and_process_images fonksiyonu, veriyi okur, her bir resmi 128x128 boyutlarına indirger ve normalizasyon yapar.
### -> Normalizasyon işlemi, her pikselin değerini 0-255 aralığından 0-1 aralığına çeker. Bu, modelin daha hızlı ve stabil şekilde öğrenmesini sağlar.

# **📊  2.3 Veriyi Eğitim ve Test Seti Olarak Ayırma**

#### 1️⃣ Etiket Kodlama 🏷️
####    Önce etiketleri (y) sayısal değerlere dönüştürdük:

##### 🔄 LabelEncoder: Her sınıfa bir numara atar. (Mesela: "Fil" = 0, "Tilki" = 1)
##### 🧮 to_categorical: Bu numaraları one-hot formata dönüştürür. (Yani: [1, 0, 0] gibi vektörler)


**🎯 Verilerin %70’i eğitim setine, %30’u test setine ayrıldı.
Bu sayede model hem öğreniyor hem de öğrendiklerini test edebiliyoruz. 📊**

# Analiz
Hayvan sınıflarından belirli bir alt küme seçildi. Toplamda 10 farklı sınıf ve her sınıf için 650 görüntü alındı.
Görüntüler, hedef boyut olan 128x128'e yeniden boyutlandırıldı ve normalizasyon işlemi yapıldı.

## Veri Seti
Bu listede, üzerinde işlem yapılacak olan 10 farklı hayvan sınıfı belirtilmiştir. Her bir sınıfın 650 örneği alınacaktır.
#### Veri seti klasörlerinin yolları
``source_dir = "/kaggle/input/animals-with-attributes-2/Animals_with_Attributes2/JPEGImages"``
#### Veri seti klasörünün yolu
``target_dir = "FilteredImages"  # Filtrelenmiş verilerin kaydedileceği yol``

#### Kullanılacak sınıflar
``selected_classes = ["collie", "dolphin", "elephant", "fox", "moose", "rabbit", "sheep", "squirrel", "giant+panda", "polar+bear"]``
``images_per_class = 650``
# **🔄 3- Veri Setinin Ayrılması ve Veri Artırımı**

####  Veri Artırma ve Eğitim:

##### 📈 Daha Fazla Veri Simülasyonu: Döndürme, kaydırma, kırpma gibi tekniklerle veri artırma yapıldı.
##### 🏋️ Eğitim ve doğrulama setleriyle modelin öğrenmesi sağlandı.

## Manipülasyonlar
Test verisi üzerinde çeşitli manipülasyonlar yapıldı (örneğin, parlaklık değişiklikleri). Bu manipülasyonlar, modelin doğruluğunu etkileyip etkilemediği konusunda test edildi.
Ayrıca, renk sabitleme algoritması (Gray World) uygulandı ve test seti üzerinde doğruluk oranları tekrar hesaplandı.
# **💡 4- CNN Modelinin Oluşturulması ve Eğitilmesi**

**🔍 Modelin Yapısı:**
####     Input Layer (Giriş Katmanı): Modelin giriş şekli (128, 128, 3) olarak belirlenmiş, yani 128x128 boyutlarında ve 3 renk kanalına sahip (RGB) görüntüler kullanılacak.
####     Conv2D: 32 adet 3x3 boyutunda konvolüsyonel filtre kullanılarak, görsellerin temel özellikleri çıkarılır.
####     MaxPooling2D: 2x2 boyutunda havuzlama yapılır, bu işlem görselin boyutunu küçültür ve özelliklerin daha kompakt bir temsiline yardımcı olur.
####     Flatten: Konvolüsyonel ve havuzlama katmanlarından çıkan veriyi bir vektöre dönüştürür.
####     Dense (128): Tam bağlantılı katman, 128 nöron ile öğrenmeye devam eder.
####     Dense (10): Çıktı katmanı, 10 sınıf için softmax aktivasyonu kullanarak her bir sınıfa ait olasılıkları hesaplar.





# ⚙️ Modeli Derleme:

**Modelin eğitimi için optimizer, loss fonksiyonu ve metrikler belirlenir.**

#### Burada kullanılan Adam optimizer, öğrenme oranı 0.001 ile ayarlanmış, modelin öğrenmesini hızlandırmak ve stabil tutmak için yaygın olarak tercih edilir. ``categorical_crossentropy``, çok sınıflı sınıflandırma problemleri için kayıp fonksiyonudur.
# 🖼️5 -Model Performansını Görselleştirme

Bu kod, modelin eğitim süreci boyunca elde ettiği doğruluk (accuracy) değerlerini görselleştirir. Eğitim ve doğrulama doğruluğunun değişimini grafikte göstererek modelin nasıl geliştiğini ve doğrulama setiyle nasıl performans gösterdiğini izlememizi sağlar.

**📊 Eğitim ve Doğrulama Doğruluğu:**
Eğitim Doğruluğu (Training Accuracy): Modelin eğitim verileri üzerinde ne kadar başarılı olduğunu gösterir. Eğitim doğruluğu arttıkça model öğrenmeye devam eder.

**Doğrulama Doğruluğu (Validation Accuracy): Modelin test verileri üzerinde ne kadar iyi performans gösterdiğini izler. Doğrulama doğruluğunun artması, modelin genelleme yeteneğinin geliştiğini gösterir.**

**📈 Grafik Yorumu:**
Eğitim doğruluğu arttıkça, modelin eğitim setinde ne kadar doğru tahminlerde bulunduğu izlenir.
Doğrulama doğruluğu, modelin eğitim verilerinin dışında ne kadar iyi genelleyebildiğini gösterir. Eğer doğrulama doğruluğu, eğitim doğruluğuyla paralel artıyorsa, model düzgün bir şekilde öğreniyor demektir.
🔑 İçgörü: Eğitim doğruluğu çok yüksek ancak doğrulama doğruluğu düşüyorsa, model overfitting (aşırı öğrenme) yapıyor olabilir. Grafik, bu tür durumların erken tespiti için yararlıdır.
# 🎨**7- Manipüle Edilmiş Test Seti ile Modeli Test Etme**

#### Bu adımda, daha önce ışık koşullarını manipüle ettiğimiz test seti kullanılarak modelin performansı yeniden değerlendirilmiştir. Manipülasyon, görsellerin parlaklık ve kontrast gibi özelliklerini değiştirerek, modelin değişen ışık koşullarına karşı ne kadar sağlam olduğunu ölçmek için yapıldı.


### 🔍 İşlem ve Sonuç:

#####      Manipülasyon: Test setindeki her bir görselin parlaklık ve kontrast ayarları değiştirildi, böylece modelin görsellerdeki bu tür değişikliklere adaptasyon yeteneği test edildi.
#### Manipülasyon fonksiyonu
``def get_manipulated_images(images):
  pass``
kullanarak Farklı Işık Manipülasyonları ile test edildi.
#####      Test: Manipüle edilmiş veri setiyle modelin doğruluk oranı değerlendirildi. Bu işlem, modelin çevresel faktörlerden nasıl etkilendiğini gözler önüne serer.



###     🎯 Amaç: Modelin sadece temiz veriyle değil, aynı zamanda gerçek dünyada karşılaşabileceği farklı ışık koşullarında da doğru sonuçlar verip vermediğini test etmekti. Bu sayede, modelin genelleme yeteneği daha da güçlendirildi.


####    📊 Sonuç: Manipüle edilmiş test seti ile yapılan test, modelin ışık değişimlerine karşı gösterdiği dayanıklılığı belirler.

# 💖 **8- Renk Sabitliği Algoritması Uygulama ve Test Etme**

### 📊 **Model Doğruluk Sonuçlarının Karşılaştırılması**

Bu aşamada, modelin farklı veri kümeleri üzerinde gösterdiği performanslar karşılaştırıldı:

#### **Orijinal Test Seti**
- **Açıklama**: Model, herhangi bir manipülasyon yapılmadan, orijinal test verisi üzerinde değerlendirildi.
- **Amaç**: Modelin temel doğruluğunu ölçmek.
- **Sonuç**: Orijinal test seti doğruluğu: **59.85%**.

#### **Manipüle Edilmiş Test Seti (Işık Koşulları Değişimi)**
- **Açıklama**: Test verilerine ışık koşulları manipülasyonu uygulandı, örneğin parlaklık ve kontrast değiştirildi.
- **Amaç**: Modelin manipülasyonlara karşı dayanıklılığını test etmek.
- **Sonuç**: Manipüle edilmiş test seti doğruluğu: **9.13%**.

#### **Renk Sabitliği Uygulanmış Test Seti**
- **Açıklama**: Manipüle edilmiş test setine Gray World algoritması ile renk sabitliği uygulandı. Bu işlem, renk dengesizliklerini düzelterek görsellerin daha tutarlı hale getirilmesini sağlar.
- **Amaç**: Manipüle edilmiş görsellerin renk dengesini iyileştirerek model doğruluğunu artırmak.
- **Sonuç**: Renk sabitliği uygulanmış test seti doğruluğu: **9.13%**.


## Çıkarım
Modelin Doğruluğu: Eğitim süreci sonunda modelin doğruluğu arttı ve %60'lar seviyesine ulaştı.

# 📌 **Model Test Edilmesi:**

####  ** Sonuçlar**:
- **Gözlemler**: Orijinal test seti ile manipüle edilmiş test seti arasında doğrulukta büyük bir düşüş gözlemlendi.
- **İyileştirme**: Renk sabitliği uygulanması, doğruluk oranını iyileştirmedi. Manipülasyon sonrası performans düşük kaldı.
- **Genel Yorum**: Modelin, özellikle ışık koşullarındaki değişikliklere karşı çok duyarlı olduğu, ancak renk sabitliği gibi düzenlemelerin doğruluğu artırmak için yeterli olmadığı söylenebilir.

Bu analiz, modelin çevresel faktörlere karşı ne kadar duyarlı olduğunu ve manipülasyonlara karşı nasıl performans gösterdiğini ortaya koymaktadır.
![image](https://github.com/user-attachments/assets/2fe5ac23-e9da-499d-8844-ca094b554120)

### Sonuç
Orijinal Test Seti Doğruluğu: 59.85%
Manipüle Edilmiş Test Seti Doğruluğu: 9.13%
Renk Sabitliği Uygulanmış Test Seti Doğruluğu: 9.13%



