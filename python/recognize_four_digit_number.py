import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


#画像の読み込みとリサイズ（画像を横の幅400・高さ100ピクセルにリサイズ）

img = cv2.imread('4.png') # ここに数字画像認識をしたい画像を入れます。事前に4等分を考慮して撮影した画像を利用
img = cv2.resize(img, (400,  100)) # サイズが調整されていない画像を入れた場合のエラー予防
#img = cv2.resize(img, (800,  200))  # 画像を横の幅800・高さ200ピクセルにリサイズするプログラム例


# 画像のトリミング [y1:y2, x1:x2]  （img = cv2.resize(img, (400,  100)) を使った場合のプログラム例

data1 = img[0:100, 0:100]     #yの範囲（縦）が0〜100・xの範囲（横）が0〜100までをトリミング
data2 = img[0:100, 100:200] #yの範囲（縦）が0〜100・xの範囲（横）が100〜200までをトリミング
data3 = img[0:100, 200:300] #yの範囲（縦）が0〜100・xの範囲（横）が200〜300までをトリミング
data4 = img[0:100, 300:400] #yの範囲（縦）が0〜100・xの範囲（横）が300〜400までをトリミング

#（img = cv2.resize(img, (800,  200)) を使った場合のプログラム例
#data1 = img[0:200, 0:200]     #yの範囲（縦）が0〜200・xの範囲（横）が0〜200までをトリミング
#data2 = img[0:200, 200:400] #yの範囲（縦）が0〜200・xの範囲（横）が200〜400までをトリミング
#data3 = img[0:200, 400:600] #yの範囲（縦）が0〜200・xの範囲（横）が400〜600までをトリミング
#data4 = img[0:200, 600:800] #yの範囲（縦）が0〜200・xの範囲（横）が600〜800までをトリミング


# トリミングした画像を保存

cv2.imwrite('data1.png', data1)
cv2.imwrite('data2.png', data2)
cv2.imwrite('data3.png', data3)
cv2.imwrite('data4.png', data4)


# 学習済みモデルの読み込み

model = load_model('model_color.h5')   # ここに学習済みモデルを入れます
                                                      # 学習済みモデルの画像のカラー設定には「モノクロ・グレースケール」「カラー」があります。

# 画像の各種設定

folder = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']  # 空白の表示に対応させるため、blankのところを「' '」で空白に設定
image_width = 28        # 使用する学習済みモデルと同じwidth（横幅）を指定
image_height = 28       # 使用する学習済みモデルと同じheight（縦の高さ）を指定
color_setting = 3          # 学習済みモデルと同じ画像のカラー設定にする。モノクロ・グレースケールの場合は「1」。カラーの場合は「3」 。
cv2_color_setting = 1   # 学習済みモデルと同じ画像のカラー設定にする。cv2.imreadではモノクロ・グレースケールの場合は「0」。カラーの場合は「1」 。


# 4桁目の画像の数値リスト化（データ化）と予測

gazou = cv2.imread('data1.png', cv2_color_setting)
gazou = cv2.resize(gazou, (image_width, image_height))
suuti = gazou.reshape(image_width, image_height, color_setting).astype('float32')/255
n1 = model.predict(np.array([suuti]))

n1 = n1[0]


# 3桁目の画像の数値リスト化（データ化）と予測

gazou = cv2.imread('data2.png', cv2_color_setting)
gazou = cv2.resize(gazou, (image_width, image_height))
suuti = gazou.reshape(image_width, image_height, color_setting).astype('float32')/255
n2 = model.predict(np.array([suuti]))

n2 = n2[0]


# 2桁目の画像の数値リスト化（データ化）と予測

gazou = cv2.imread('data3.png', cv2_color_setting)
gazou = cv2.resize(gazou, (image_width, image_height))
suuti = gazou.reshape(image_width, image_height, color_setting).astype('float32')/255
n3 = model.predict(np.array([suuti]))

n3 = n3[0]


# 1桁目の画像の数値リスト化（データ化）と予測

gazou = cv2.imread('data4.png', cv2_color_setting)
gazou = cv2.resize(gazou, (image_width, image_height))
suuti = gazou.reshape(image_width, image_height, color_setting).astype('float32')/255
n4 = model.predict(np.array([suuti]))

n4 = n4[0]


#結果の表示（個別に認識した結果を、4桁目から順番に並べています。予測確率が高い数字を表示させています）

print('7セグメント連続数字画像の認識結果（予測結果）：\n\n', folder[n1.argmax()], folder[n2.argmax()], folder[n3.argmax()], folder[n4.argmax()])


print('\n\n【今回認識した元の画像】')


# 元の画像を4分割でトリミングした画像の表示：複数の画像を表示させる
img_list=[data1, data2, data3, data4]
for bangou, imagenamae in enumerate(img_list):
    plt.subplot(2, 4, bangou+1) #(行数,列数,何番目に画像を表示させるか)
    plt.axis("off") #画像の軸をオフ
    plt.title('data' +str(bangou+1))
    plt.imshow(cv2.cvtColor(imagenamae, cv2.COLOR_BGR2RGB))
