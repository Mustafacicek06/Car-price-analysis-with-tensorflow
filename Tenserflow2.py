# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 11:46:18 2020

@author: GRUNDİG
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf

dataFrame = pd.read_excel("merc.xlsx")

dataFrame.head()

dataFrame.describe()

# null olanların toplamını veriyor
dataFrame.isnull().sum()

# figürün büyüklüğü 
plt.figure(figsize=(8,6))
# grafiğin son kısımlarını cıkarmamız lazım veriyi düzgün işleyebilmek için
# distplot nasıl dağıldığını göster
sb.distplot(dataFrame["price"])
# kaç tane dağıldığını göster

sb.countplot(dataFrame["year"])
# correlation (ilişki)
dataFrame.corr()

# fiyatın diğer değişkenlerle ilişkisini görmek için
dataFrame.corr()["price"].sort_values()

# x = futures (özellik ) , y gitmek istediğiniz özellik
sb.scatterplot(x="year",y="price",data=dataFrame)

# ascending(yükselen) = false en yüksek fiyattan başla sıralamaya
dataFrame.sort_values("price",ascending=False).head(22)

# veri kümesinin %99 unu alırsak yine grafik değişmeyeceğinden
# en yüksek fiyatlıları çıkarıyoruz.
len(dataFrame) * 0.01

# en baştaki 131 den sonrasını sırala işlemi 
yenidataFrame = dataFrame.sort_values("price",ascending=False).iloc[131:]

yenidataFrame.describe()

plt.figure(figsize=(7,5))
sb.distplot(yenidataFrame["price"])

# yıla göre grupla ve fiyat ortalamasını al 
dataFrame.groupby("year").mean()["price"]

yenidataFrame.groupby("year").mean()["price"]

dataFrame = yenidataFrame

dataFrame.describe()

dataFrame = dataFrame[dataFrame.year != 1970]

dataFrame.groupby("year").mean()["price"]

# sayı değeri olmadığı için çıkarıyoruz regression da hata vermemesi için
dataFrame = dataFrame.drop("transmission",axis=1)

# numpy dizisine cevirdik
y= dataFrame["price"].values

x= dataFrame.drop("price",axis=1).values

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=10)

len(x_train)
len(x_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
# ulaşmak istediğimiz değeri scale etmemize gerek yok işlemeyeceğimiz için

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# (9090, 5) 5 layerden başlatmamız bizim için iyi olur
x_train.shape

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))



# çıkış katmanı
model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")


# batch_size verileri parti parti vermek demektir
# bir anda tüm veriyi vermek nöronları yorabilir
# epochs yüksek verilirse overfiting yaşayabiliriz
# en uygun ağırlıkları bulmaya calısır 
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=300,epochs=300)

lossData = pd.DataFrame(model.history.history)

lossData.head()

# veri cizgileri sonlara doğru eğer birbirinden uzaklaşırsa
# overfiting yaşanıyor demektir
lossData.plot()


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

tahminArray = model.predict(x_test)

tahminArray


# 3195 pound gerçek değerden sapma tespit ettik bu da
# %13 hata payı demektir ve büyük bi hata payı bu verilere göre
# layerları attırmak , epochsu arttırmak veya verileri temizlemek
# nöronları arttırmak çözüm olabilir 
mean_absolute_error(y_test,tahminArray)

# lineer regression bulmus gibi gözüküyoruz
plt.scatter(y_test,tahminArray)
plt.plot(y_test,y_test,"g-*")

dataFrame.iloc[2]

newcarSeries = dataFrame.drop("price",axis=1).iloc[2]

# -1,5 5 tane özelliği olduğu için böyle yazdım
# 1 tane örnek içerseydi 1,-1 olacaktı
newcarSeries = scaler.transform(newcarSeries.values.reshape(-1,5))


model.predict(newcarSeries)

testdf = pd.DataFrame(y_test)
traindf = pd.DataFrame(y_train)

test_train = pd.concat([testdf,traindf],axis=1)

print(test_train)













































