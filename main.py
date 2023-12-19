import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV với phân tách ";"
data = pd.read_csv("DuLieuYKhoa.csv", sep=";")

# Chọn các features cần sử dụng
X = data[['TUOI', 'CHOLESTEROL', 'TRIGLYCERIDE', 'HA', 'BMI']]

# Chuẩn hóa dữ liệu
X = (X - X.mean()) / X.std()

# Thực hiện phân tích PCA với số thành phần chính là 5
pca = PCA(n_components=5)
principal_components = pca.fit_transform(X)

# Lấy các trọng số của các thành phần chính
weights = pca.components_[0]

#  Tính nguy cơ bệnh Gout dựa trên PCA
data['NGUYCOGOUT_PCA'] = np.dot(X, weights)

# Tính độ sai số
data['Sai So'] = data['NGUYCOGOUT'] - data['NGUYCOGOUT_PCA']

# So sánh giá trị NGUYCOGOUT và NGUYCOGOUT_PCA
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['NGUYCOGOUT'], label='Giá trị thực tế', alpha=0.5)
plt.scatter(data.index, data['NGUYCOGOUT_PCA'], label='Giá trị dự đoán với PCA', alpha=0.5)
plt.xlabel('Mẫu dữ liệu')
plt.ylabel('Tỷ lệ Gout')
plt.title(f'So sánh giá trị thực tế và giá trị dự đoán với PCA \n Độ sai số trung bình:{data["Sai So"].mean()}')
plt.legend()
plt.grid(True)
plt.show()

# Thống kê tỷ lệ Gout theo các nhóm độ tuổi
age_groups = ['0-24', '25-50', '50+']
gout_rates = []

for age_group in age_groups:
    if age_group == '0-24':
        subset = data[data['TUOI'] <= 24]
    elif age_group == '25-50':
        subset = data[(data['TUOI'] > 24) & (data['TUOI'] <= 50)]
    else:
        subset = data[data['TUOI'] > 50]

    gout_rate = subset['NGUYCOGOUT'].mean()
    gout_rates.append(gout_rate)

# Vẽ biểu đồ tỷ lệ Gout
plt.figure(figsize=(8, 6))
plt.bar(age_groups, gout_rates)
plt.xlabel('Nhóm Tuổi')
plt.ylabel('Tỷ lệ Gout')
plt.title('Tỷ lệ Gout theo các nhóm độ tuổi')
plt.grid(axis='y')
plt.show()

# In trọng số của các biến độc lập
print("Trọng số của các biến độc lập:")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {weights[i]}")

# Vẽ biểu đồ trọng số của các biến độc lập
plt.figure(figsize=(8, 6))
plt.bar(X.columns, weights)
plt.xlabel('Biến độc lập')
plt.ylabel('Trọng số')
plt.title('Trọng số của các biến độc lập trong PCA')
plt.grid(axis='y')
plt.show()
