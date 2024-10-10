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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score




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

# oulier_datapoints = detect_outlier(kolom_outlier)
# print(outlier_datapoints)

# Menampilkan outlier dari setiap kolom
print("Outliers per column:")
for outlier in outliers:
    print(outlier)

print("jumlah Outliers per column:")
for outlier in outliers:
    for key, value in outlier.items():
        print(f"Kolom {key}: {len(value)}")

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
x = clean_data.iloc[:,1:20].values
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

# INISIALISASI MODEL (RandomForest)
clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# MELATIH MODEL DENGAN DATA TRAINING
clf.fit(x_train, y_train)

# MELAKUKAN PREDIKSI
y_pred = clf.predict(x_test)

# MENGHITUNG AKURASI, PRECISION, RECALL, F1-SCORE
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

# MENAMPILKAN METRIK
print("Evaluation Metrics".center(75, "="))
print(f"Akurasi  : {accuracy:.4f}")
print(f"Presisi  : {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("=" * 75)

# MENAMPILKAN LAPORAN KLASIFIKASI
print("Classification Report".center(75, "="))
print(classification_report(y_test, y_pred, zero_division=0))

# # MENAMPILKAN CONFUSION MATRIX
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(cm).plot(cmap='Blues')
# plt.title("Confusion Matrix")
# plt.show()


cm = confusion_matrix(y_test, y_pred)

# Menampilkan Confusion Matrix
print("Confusion Matrix:")
print(cm)

# Menghitung TP, FP, FN, dan TN untuk setiap kelas
for i in range(len(cm)):
    tp = cm[i, i]  # True Positive untuk kelas ke-i
    fp = cm[:, i].sum() - tp  # False Positive untuk kelas ke-i
    fn = cm[i, :].sum() - tp  # False Negative untuk kelas ke-i
    tn = cm.sum() - (tp + fp + fn)  # True Negative untuk kelas ke-i

    print(f"\nClass {i}:")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Negative (TN): {tn}")


    # INISIALISASI MODEL (Decision Tree)
dt_model = DecisionTreeClassifier(random_state=42)

# MELATIH MODEL DENGAN DATA TRAINING
dt_model.fit(x_train, y_train)

# MELAKUKAN PREDIKSI MENGGUNAKAN DECISION TREE
y_pred_dt = dt_model.predict(x_test)

# MENGHITUNG AKURASI, PRECISION, RECALL, F1-SCORE UNTUK DECISION TREE
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro', zero_division=0)
recall_dt = recall_score(y_test, y_pred_dt, average='macro', zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average='macro', zero_division=0)

# MENAMPILKAN METRIK UNTUK DECISION TREE
print("Evaluation Metrics (Decision Tree)".center(75, "="))
print(f"Akurasi  : {accuracy_dt:.4f}")
print(f"Presisi  : {precision_dt:.4f}")
print(f"Recall   : {recall_dt:.4f}")
print(f"F1 Score : {f1_dt:.4f}")
print("=" * 75)

# MENAMPILKAN LAPORAN KLASIFIKASI UNTUK DECISION TREE
print("Classification Report (Decision Tree)".center(75, "="))
print(classification_report(y_test, y_pred_dt, zero_division=0))


# Mendapatkan class_names dari data target
class_names = [str(cls) for cls in np.unique(y_train)]

# # Visualisasi Decision Tree
# plt.figure(figsize=(15, 10))
# plot_tree(dt_model, filled=True, feature_names=clean_data.columns[1:], class_names=class_names)
# plt.title('Decision Tree')
# plt.show()

# Menghitung metrik untuk Random Forest
accuracy_rf = accuracy_score(y_test, y_pred)
precision_rf = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_rf = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_rf = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Menampilkan metrik untuk Random Forest
print("Metrik Kinerja (Random Forest)".center(75, "="))
print(f"Akurasi  : {accuracy_rf:.4f}")
print(f"Presisi  : {precision_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"F1 Score  : {f1_rf:.4f}")
print("=" * 75)

# Menghitung metrik untuk Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro', zero_division=0)
recall_dt = recall_score(y_test, y_pred_dt, average='macro', zero_division=0)
f1_dt = f1_score(y_test, y_pred_dt, average='macro', zero_division=0)

# Menampilkan metrik untuk Decision Tree
print("Metrik Kinerja (Decision Tree)".center(75, "="))
print(f"Akurasi  : {accuracy_dt:.4f}")
print(f"Presisi  : {precision_dt:.4f}")
print(f"Recall    : {recall_dt:.4f}")
print(f"F1 Score  : {f1_dt:.4f}")
print("=" * 75)


def evaluate_model(y_true, y_pred, model_name):  
    """  
    Menghitung dan menampilkan metrik evaluasi untuk model yang diberikan.  
    
    Parameters:  
    y_true : array-like  
        Nilai sebenarnya dari kelas.  
    y_pred : array-like  
        Nilai yang diprediksi oleh model.  
    model_name : str  
        Nama model (untuk tampilan).  
    """  
    accuracy = accuracy_score(y_true, y_pred)  
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)  
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)  
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)  
    
    print(f"Evaluation Metrics ({model_name})".center(75, "="))  
    print(f"Akurasi  : {accuracy:.4f}")  
    print(f"Presisi  : {precision:.4f}")  
    print(f"Recall   : {recall:.4f}")  
    print(f"F1 Score : {f1:.4f}")  
    print("=" * 75)  

# Menghitung metrik untuk Random Forest  
evaluate_model(y_test, y_pred, "Random Forest")  

# Menghitung metrik untuk Decision Tree  
evaluate_model(y_test, y_pred_dt, "Decision Tree")


