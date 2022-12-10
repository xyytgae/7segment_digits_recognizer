#1 ライブラリのインポート等

from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np


#2 各種設定

recognise_image = '47.png' #ここを変更。画像認識したい画像ファイル名。（実行前に認識したい画像ファイルを1つアップロードしてください）
folder = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '']  # 空白の表示に対応させるため、blankのところを「' '」で空白に設定
image_width = 28  # ここを変更。利用する学習済みモデルの横の幅のピクセル数と同じにする。
image_height = 28 # ここを変更。利用する学習済みモデルの縦の高さのピクセル数と同じにする。
color_setting = 1    # ここを変更。利用する学習済みモデルのカラー形式と同じにする。「1」はモノクロ・グレースケール。「3」はカラー。
cv2_color_setting = 0  # ここを変更。学習済みモデルと同じ画像のカラー設定にする。cv2.imreadではモノクロ・グレースケールの場合は「0」。カラーの場合は「1」 。

#3 各種読み込み

model = load_model('model.h5')  #ここを変更。読み込む学習済みモデルを入れます
#model = load_model('keras_cnn_7segment_digits_gray28*28_model.h5')  #モノクロ・グレー形式の学習済みモデルを読み込む例：color_setting = 1 の学習済みモデルを使う場合
#model = load_model('keras_cnn_7segment_digits_color28*28_model.h5')  #カラー形式の学習済みモデルを読み込む例：color_setting = 3 の学習済みモデルを使う場合


#4 画像の表示・各種設定等

img = cv2.imread(recognise_image, cv2_color_setting)
img = cv2.resize(img, (image_width, image_height))
plt.imshow(img)
# plt.gray()  #ここを変更。カラーの場合は「plt.gray()」を消す。モノクロ・グレースケールの場合は「plt.gray()」が無いと変な色になります。
plt.show()

img = img.reshape(image_width, image_height, color_setting).astype('float32')/255


#5 予測と結果の表示等

prediction = model.predict(np.array([img]))
result = prediction[0]

for i, accuracy in enumerate(result):
  print('画像認識AIは「', folder[i], '」の確率を', int(accuracy * 100), '% と予測しました。')

print('-------------------------------------------------------')
print('予測結果は、「', folder[result.argmax()],'」です。')
print(' \n\n　＊　「確率精度が低い画像」や、「間違えた画像」を再学習させて、オリジナルのモデルを作成してみてください。')
print(' \n　＊　「間違えた画像」を数枚データセットに入れるだけで正解できる可能性が向上するようでした。')
print(' \n　＊　「0」と「」（blank）の認識で迷う傾向があるようです。')