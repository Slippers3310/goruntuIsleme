# Aygaz Görüntü İşleme Bootcamp
Bu projede, görüntü işleme ve derin öğrenme tekniklerini kullanarak, çeşitli hayvan sınıflarını sınıflandırmaya yönelik bir Convolutional Neural Network (CNN) modeli oluşturulmuştur. Projede kullanılan ana araçlar arasında veri işleme, model eğitimi ve görselleştirme bulunmaktadır.

## Kullandığım kütüphaneler :
  NumPy: Veri işleme ve matematiksel hesaplamalar için kullanıldı.
  OpenCV: Görüntü işleme ve görsellerin boyutlandırılması, normalizasyonu için kullanıldı.
  scikit-learn: Veri kümesinin bölünmesi ve etiketleme için kullanıldı.
  TensorFlow (Keras): Derin öğrenme modeli oluşturma, eğitme ve değerlendirme işlemleri için kullanıldı.
  Matplotlib: Eğitim ve doğrulama doğruluğu ile kaybı görselleştirme için kullanıldı.

## Analiz
Hayvan sınıflarından belirli bir alt küme seçildi. Toplamda 10 farklı sınıf ve her sınıf için 650 görüntü alındı.
Görüntüler, hedef boyut olan 128x128'e yeniden boyutlandırıldı ve normalizasyon işlemi yapıldı.

## Manipülasyonlar
Test verisi üzerinde çeşitli manipülasyonlar yapıldı (örneğin, parlaklık değişiklikleri). Bu manipülasyonlar, modelin doğruluğunu etkileyip etkilemediği konusunda test edildi.
Ayrıca, renk sabitleme algoritması (Gray World) uygulandı ve test seti üzerinde doğruluk oranları tekrar hesaplandı.


## Çıkarım
Modelin Doğruluğu: Eğitim süreci sonunda modelin doğruluğu arttı ve %60'lar seviyesine ulaştı.
