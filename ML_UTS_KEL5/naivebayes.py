import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import sklearn.model_selection as model_selection
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             f1_score, ConfusionMatrixDisplay,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

dataframe = pd.read_csv("Leukemia_GSE9476.csv")

data = dataframe[['type','1007_s_at','1053_at','117_at','121_at','1255_g_at',
                '1294_at','1316_at','1320_at','1405_i_at','1431_at','1438_at',
                '1487_at','1494_f_at','1598_g_at','160020_at','1729_at','1773_at',
                '177_at','1861_at','200000_s_at']]

print("data awal".center(75,"="))
print(data)
print("==========================================")

# PRE PROCESSING
    # MISSING VALUE
print("pengecekan missing value".center(75,"="))
print(data.isnull().sum())
print("============================================================")

    # OUTLIER
kolom_outlier = data[['1007_s_at','1053_at','117_at','121_at','1255_g_at',
                '1294_at','1316_at','1320_at','1405_i_at','1431_at','1438_at',
                '1487_at','1494_f_at','1598_g_at','160020_at','1729_at','1773_at',
                '177_at','1861_at','200000_s_at']]

outliers = []

def detect_outlier(data_1):
    threshold = 3
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    col_outliers = []  # Outlier per kolom
    for y in data_1:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            col_outliers.append(y)  # Simpan nilai outlier
    return col_outliers

# Looping untuk setiap kolom di kolom_outlier
for column in kolom_outlier.columns:
    col_data = kolom_outlier[column].values
    col_outliers = detect_outlier(col_data)
    outliers.append({column: col_outliers})

# Menampilkan outlier dari setiap kolom
print("Outliers per column:")
for outlier in outliers:
    print(outlier)

# MENGHAPUS OUTLIER
    # Ambil semua indeks yang mengandung outlier
outlier_indices = []

for column in kolom_outlier.columns:
    col_data = kolom_outlier[column].values
    threshold = 3
    mean_1 = np.mean(col_data)
    std_1 = np.std(col_data)

    for i, y in enumerate(col_data):
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outlier_indices.append(i)  # Simpan indeks outlier

    #Menghapus baris berdasarkan indeks outlier
outlier_indices = list(set(outlier_indices))  # Pastikan indeks unik
clean_data = data.drop(index=outlier_indices)  # Hapus baris berdasarkan indeks

print("Data tanpa outliers".center(75, "="))
print(clean_data)

print("Outliers per column:")
for outlier in outliers:
    for key, value in outlier.items():
        print(f"Kolom {key}: {len(value)}")


# NORMALISASI / SCALING DATA (KITA DITENTUKAN UNTUK MENGGUNAKAN SIMPLE FEATURE SCALING)

clean_data['1007_s_at'] = clean_data['1007_s_at'] / clean_data['1007_s_at'].max()
clean_data['1053_at'] = clean_data['1053_at'] / clean_data['1053_at'].max()
clean_data['117_at'] = clean_data['117_at'] / clean_data['117_at'].max()
clean_data['121_at'] = clean_data['121_at'] / clean_data['121_at'].max()
clean_data['1255_g_at'] = clean_data['1255_g_at'] / clean_data['1255_g_at'].max()
clean_data['1294_at'] = clean_data['1294_at'] / clean_data['1294_at'].max()
clean_data['1316_at'] = clean_data['1316_at'] / clean_data['1316_at'].max()
clean_data['1320_at'] = clean_data['1320_at'] / clean_data['1320_at'].max()
clean_data['1405_i_at'] = clean_data['1405_i_at'] / clean_data['1405_i_at'].max()
clean_data['1431_at'] = clean_data['1431_at'] / clean_data['1431_at'].max()
clean_data['1438_at'] = clean_data['1438_at'] / clean_data['1438_at'].max()
clean_data['1487_at'] = clean_data['1487_at'] / clean_data['1487_at'].max()
clean_data['1494_f_at'] = clean_data['1494_f_at'] / clean_data['1494_f_at'].max()
clean_data['1598_g_at'] = clean_data['1598_g_at'] / clean_data['1598_g_at'].max()
clean_data['160020_at'] = clean_data['160020_at'] / clean_data['160020_at'].max()
clean_data['1729_at'] = clean_data['1729_at'] / clean_data['1729_at'].max()
clean_data['1773_at'] = clean_data['1773_at'] / clean_data['1773_at'].max()
clean_data['177_at'] = clean_data['177_at'] / clean_data['177_at'].max()
clean_data['1861_at'] = clean_data['1861_at'] / clean_data['1861_at'].max()
clean_data['200000_s_at'] = clean_data['200000_s_at'] / clean_data['200000_s_at'].max()

print("DATA AFTER SCALING (SIMPLE FEATURE)")
print(clean_data)
print("====================================================")

# PENGELOMPOKAN KOLOM
    # encode kolom yang bukan numerik (karena train test split hanya dapat menerima & memproses numerik)
labelencoder = LabelEncoder()
clean_data['type'] = labelencoder.fit_transform(clean_data['type'])

print("GROUPING COLUMN".center(75,"="))
x = clean_data.iloc[:,1:].values
y = clean_data.iloc[:,0].values
print("DATA VARIABEL".center(75,"="))
print(x)
print("DATA CLASS (OUTPUT)".center(75,"="))
print(y,"\n")

# PEMBAGIAN DATA TRAINING DAN TESTING

print("SPLITTING DATA (90%-10%)".center(75,"="), "\n")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)
print("instance variabel data training".center(75,"="))
print(x_train,"\n")
print("instance kelas data training".center(75,"="))
print(y_train,"\n")
print("instance variabel data testing".center(75,"="))
print(x_test,"\n")
print("instance kelas data testing".center(75,"="))
print(y_test,"\n")

#pemodelan naive bayes 
print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test) 
accuracy_nb=round(accuracy_score(y_test,y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
print("instance prediksi naive bayes:")
print(y_pred,"\n")

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("LAPORAN KLASIFIKASI DENGAN NAIVE BAYES".center(75,"="))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f"accuracy = {accuracy}")
print(f"precision = {precision}")
print(f"recall = {recall}")
print(f"f1 score = {f1}")

@app.route("/")
def home():
    return render_template("landingpage.html")

@app.route("/deteksi")
def deteksi():
    return render_template("deteksi.html")

@app.route("/hasil", methods=['GET', 'POST'])
def hasil():
    if request.method == 'POST':
        # Collect data from the form
        new_record = []
        for col in data.columns[1:]:
            value = float(request.form[col])
            new_record.append(value)

        new_record = [new_record]

        # Normalize the input values using the same scaling method
        new_record_scaled = new_record.copy()
        for i, col in enumerate(data.columns[1:]):
            new_record_scaled[0][i] = new_record[0][i] / data[col].max()

        # Predict the leukemia type
        # new_record_scaled = np.array([input_scaled])
        predicted_type = gaussian.predict(new_record_scaled)
        predicted_label = labelencoder.inverse_transform(predicted_type)

        return render_template('hasil.html', predicted_type=predicted_label[0])


if __name__ == '__main__':
    app.run(debug=True)