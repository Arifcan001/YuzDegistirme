# YuzDegistirme
Bu proje, InsightFace kütüphanesini ve Gradio arayüzünü kullanarak videolardaki yüzleri yüksek doğrulukla değiştiren bir uygulamadır. Mobil uygulama için optimize edilmiştir.

**Kurulum**

1. Depoyu Klonlayın

git clone [https://github.com/Arifcan001/YuzDegistirme](https://github.com/Arifcan001/YuzDegistirme).git
cd YuzDegistirme


2. Sistem Paketlerini Kurun (Linux/Ubuntu)

Uygulamanın packages.txt dosyasındaki paketlere ihtiyacı vardır.


3. Python Bağımlılıklarını Yükleyin

pip install -r requirements.txt


4. Model Dosyasını İndirin

Projenin çalışması için inswapper_128.onnx model dosyasının ana dizinde bulunması gerekir.

**Kullanım**

Uygulamayı başlatmak için:

python app.py


Arayüz üzerinden adımlar:

Kaynak Yüz: Videoya yerleştirmek istediğiniz yüzün fotoğrafını yükleyin.

Hedef Kişi: Videoda değiştirilmesini istediğiniz kişinin fotoğrafını referans olarak yükleyin.

Video: İşlem yapılacak hedef videoyu yükleyin.

Başlat: İşlemi başlatın.


**Yasal Uyarı**

Eğitim amaçlı geliştirilmiştir.

Kullanıcılar, bu aracı kullanırken yasalara ve etik kurallara uymakla yükümlüdür. Rıza dışı içerikler üretmek için kullanılması kesinlikle önerilmez ve yasal sorumluluk doğurabilir.

Oluşturulan içeriklerin kötüye kullanımından veya doğabilecek yasal sorunlardan geliştirici sorumlu tutulamaz.
