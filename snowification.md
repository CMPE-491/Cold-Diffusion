# SNOWIFICATION ADIMLARI

1. Verilen snow level oranına bağlı olarak belirli parametreler tanımlanır. Bunlar:
Karlı pikseller için Gauss dağılımının ortalama/standart sapması (base layer oluşturmak için kullanılır; Gauss gürültüsü oluşturulur.)
Zoom faktörü (oluşturulan layer belirlenen oranda zoomlanır)
Snowification için başlangıç ve bitiş eşik değerleri (her adım için bu değerler arasında lineer bir şekilde eşik değeri oluşturulur, örnek: başlangıç:0.7, bitiş:0.3, t1-eşik=0.7, t2-eşik=0.6, vs. Bu değerlerin altındaki pikseller kırpılır.)
Motion blur yarıçapı ve sigma başlangıç/bitir değerleri (her timestep  için lineer dağıtılacak şekilde değer atanır; Gaussian kernel için bu sigma değeri kullanılır.)
Parlaklık katsayısı başlangıç ve bitir değerleri (her timestep için lineer dağıtılacak şekilde değer atanır; gri tonlama resmi ve 0-1 arasında ölçeklenmiş resim için oranı belirler. fix brightness açık değilse kullanılır.)
2. Belirlenen ortalama/standart sapma değeri ve Gauss dağılımı ile snow_layer_base oluşturulur. (single_snow parametresi varsa, her resim için ayrı base oluşturulur; yoksa tümü aynı base'i kullanır.)

3. Zoom faktörüne göre zoom uygulanır ve gerekli boyut düzenlemeleri yapılır.

4. %50 ihtimalle karın düşüş yönü seçilir (dikey veya yatay).

5. Her timestep için:
    1. Base layer üzerinde, o timestep ait eşik değerine göre, bu değerden küçük pikseller sıfırlanır ve resim uygun boyuta getirilir.
    2. O timestep için belirlenen sigma ve yarıçap değerleri ile Gauss kernel oluşturulur.
    3. Bu kernel, uygun boyutlu bir motion kernel'in orta satırına yerleştirilir.
    4. Eğer karın düşüş yönü dikey seçilmişse, kernel uygun şekilde döndürülür.
    5. Seçilen yön doğrultusunda, convolution işlemi uygulanarak kar efektleri elde edilir. (Eğer single_snow parametresi varsa, yarısı yatay yarısı dikey olarak işlenir.)

6. Resim -1 ile 1 aralığından 0 ile 1 aralığına çekilir.

7. fix_brightness parametresi belirtilmişse son adıma geçilir.

8. Ardından, resim gri tonlamaya çevrilir ve (resim * 1.5 + 0.5) işlemi uygulanır. Bu, desatürasyon sağlar ve resmi aydınlatır.

9. Güçlü piksel değerlerini korumak için, orijinal resim ile yeni resim arasında torch.maximum(org, new) işlemi uygulanır.

10. O timestep  için seçilen parlaklık katsayısı ile orijinal ve yeni oluşturulan resim karıştırılarak son görüntü elde edilir.

11. Kar değerleri eklenir ve resim 0,1 aralığından -1,1 aralığına çekilerek işlem tamamlanır.
