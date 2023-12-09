## Is Problemi

# Turkiye'nin en buyuk online hizmet platformlarindan biri Armut. Hizmet verenler ile hizmet vermek isteyenleri bulusturuyor.
# Association Rule Learning ile urun tavsiye sistemi olusturuyoruz.

## Veri Seti

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih

#!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
df_ = pd.read_csv('dataset/armut_data.csv')
df = df_.copy()

# ServiceID ve CategoryId'yi '_' ile birlestirerek hizmetleri temsil edecek yeni bir degisken olusturuyoruz.
df['Hizmet'] =[str(row[1]) + '_' + str(row[2]) for row in df.values]

# Sadece ay ve yil olan bir dagisken olusturuyoruz.
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df['NewDate'] = df["CreateDate"].dt.strftime('%Y-%m')

# Burdaki sepet tanimi her bir musterinin aylik aldigi hizmetlerdir.
df['SepetID'] = [str(row[0]) + '_' + str(row[5]) for row in df.values]


# Asagidaki gibi bir sepet hizmet pivot table'i olusturuyoruz.
# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Apriori
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.01)
rules.head()

# Association Rule Learning Recommender fonksiyonu kullanarak en son 2_0 hizmetini alan bir kullaniciya hizmet onerisinde bulunalim.
def arl_recommender(rules_df, product_id, rec_count):
    sorted_rules = rules_df.sort_values('lift', ascending=False) # confidence'e gore de siralanabilir.
    recommendation_list = [] # Tavsiye edilecek hizmetler icin bos bir liste.
    for i, product in sorted_rules['antecedents'].items(): # # i: index, antecedents\product: x
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]['consequents']))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list}) # Tavsiye listesindeki tekrarlari onlemek icin.
    return recommendation_list[:rec_count] # rec_count: istenen sayiya kadar tavsiye urun getirir.

arl_recommender(rules,"2_0", 4) # ['15_1', '2_0', '25_0', '38_4']


