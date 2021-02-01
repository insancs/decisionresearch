import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('E:/DQLab/Dataset/data_retail.csv', sep=';')
# Menampilkan semua kolom tanpa terpotong
# pd.options.display.max_columns = None
pd.options.display.width = None
# Menampilkan kolom dari index ke-2 sampai akhir
print(df.iloc[:, 2:].head())
print("Informasi data:\n", df.info())

"""---------------DATA PREPARATION--------------"""
# Ubah kolom First_Transaction dan Last_Transaction menjadi datetime
df['First_Transaction'] = pd.to_datetime(df['First_Transaction'] / 1000, unit='s', origin='1970-01-01')
df['Last_Transaction'] = pd.to_datetime(df['Last_Transaction'] / 1000, unit='s', origin='1970-01-01')

# Tampilkan type data
print("\nType data :\n", df.dtypes)

# [1] Pengecekan transaksi terakhir dalam dataset
print("\n[1] Transaksi terakhir\n", max(df['Last_Transaction']))

# [2] Klasifikasi customer yang berstatus churn atau tidak dengan boolean
print("\nDF loc\n", df.loc[df['Last_Transaction'] <= '2018-08-01'])
df.loc[df['Last_Transaction'] <= '2018-08-01', 'is_churn'] = True
df.loc[df['Last_Transaction'] > '2018-08-01', 'is_churn'] = False

# [3] Hapus kolom yang tidak diperlukan
del df['no']
del df['Row_Num']
print("\n[3] Setelah drop kolom\n", df.head())

"""--------------DATA VISUALIZATION--------------"""
# [4] Kolom tahun transaksi pertama
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
# [5] Kolom tahun transaksi terakhir
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year

df_year_viz = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
print("\n", df_year_viz)
# Visualisasi
df_year_viz.plot(kind='bar')
plt.title('Graph of Customer Acquisition', color='red')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Customers')
plt.xticks(rotation=0, size=10)
plt.tight_layout()
plt.show()

# [6] Transaction by Year
df_year_count = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
print(df_year_count)
# Visualisasi
df_year_count.plot(kind='bar')
plt.title('Graph of Transaction Customer', color='red')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Transaction')
plt.xticks(rotation=0, size=10)
plt.tight_layout()
plt.show()

# [7] Average Transaction Amount by Year Tiap Produk
print("\nAverage transaction amount by Year :\n",
      df[['Year_First_Transaction', 'Average_Transaction_Amount', 'Product']].groupby(
          ['Product', 'Year_First_Transaction']).mean().reset_index())
# Visualisation
sns.pointplot(data=df.groupby(['Product', 'Year_First_Transaction']).mean().reset_index(), x='Year_First_Transaction',
              y='Average_Transaction_Amount', hue='Product')
plt.title('Average transaction amount by Year')
plt.xlabel('Year First Transaction')
plt.ylabel('Average Transaction Amount')
plt.tight_layout()
plt.show()

# [8] Melakukan pivot data dengan pivot_table
df_viz = df.pivot_table(index='is_churn', columns='Product', values='Customer_ID', aggfunc='count', fill_value=0)
print("\n[8] Proportion Churn by Product :\n", df_viz)
# Mendapatkan Proportion Churn by Product
plot_product_column = df_viz.count().sort_values().head().index
print("\n[9] Plot column:", plot_product_column)
# Plot pie chart
df_viz = df_viz.reindex(columns=plot_product_column)
df_viz.plot.pie(subplots=True, figsize=(10, 7), layout=(-1, 2), autopct='%1.0f%%', title='Proportion Churn by Product')
plt.tight_layout()
plt.show()


# [9] Kategorisasi jumlah transaksi
def func(row):
    if row['Count_Transaction'] == 1:
        val = '1. 1'
    elif row['Count_Transaction'] >= 2 and row['Count_Transaction'] <= 3:
        val = '2. 2-3'
    elif row['Count_Transaction'] >= 4 and row['Count_Transaction'] <= 6:
        val = '3. 4-6'
    elif row['Count_Transaction'] >= 7 and row['Count_Transaction'] <= 10:
        val = '4. 7-10'
    else:
        val = '5. >10'
    return val


df['Count_Transaction_Group'] = df.apply(func, axis=1)
count_trans_group = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
print("\n[9] Count Transaction Group :\n", count_trans_group)
# Visualisasi Count_Transaction_Group
count_trans_group.plot(kind='bar')
plt.title('Customer Distribution by Count Transaction Group')
plt.xlabel('Count Transaction Group')
plt.ylabel('Num of Customer')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# [10] Distribusi Kategorisasi Average_Transaction_Amount
def f(row):
    if row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <= 250000:
        val = '1. 100.000 - 250.000'
    elif row['Average_Transaction_Amount'] > 250000 and row['Average_Transaction_Amount'] <= 500000:
        val = '2.>250.000 - 500.000'
    elif row['Average_Transaction_Amount'] > 500000 and row['Average_Transaction_Amount'] <= 750000:
        val = '3.>500.000 - 750.000'
    elif row['Average_Transaction_Amount'] > 750000 and row['Average_Transaction_Amount'] <= 1000000:
        val = '4.>750.000 - 1.000.000'
    elif row['Average_Transaction_Amount'] > 1000000 and row['Average_Transaction_Amount'] <= 2500000:
        val = '5.>1.000.000 - 2.500.000'
    elif row['Average_Transaction_Amount'] > 2500000 and row['Average_Transaction_Amount'] <= 5000000:
        val = '6.>2.500.000 - 5.000.000'
    elif row['Average_Transaction_Amount'] > 5000000 and row['Average_Transaction_Amount'] <= 10000000:
        val = '7.>5.000.000 - 10.000.000'
    else:
        val = '8.>10.000.000'
    return val


df['Average_Transaction_Amount_Group'] = df.apply(f, axis=1)
avg_trans_group = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
print("\n[10] Average Transaction Group\n", avg_trans_group)
avg_trans_group.plot(kind='bar')
plt.title('Customer Distribution by Average Transaction Amount Group')
plt.xlabel('Average Transaction Amount Group')
plt.ylabel('Num of Customer')
# labels, location = plt.xticks()
# plt.xticks(labels, (labels / 10))
plt.tight_layout()
plt.show()

# [11] Pemodelan
# Feature column: Year_Diff
df['Year_Diff'] = df['Year_Last_Transaction'] - df['Year_First_Transaction']
print("\nDataset :\n", df.head())

# Feature variable
X = df[['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']]
print("Data X :\n", X.head())
y = df['is_churn']
print("Data Y :\n", y.head())

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Shape of X_train :\n", X_train, "\nShape of y_train :\n", y_train)

# # Call classifier
# logreg = LogisticRegression()
# # Fit the classifier to the training data
# logreg = logreg.fit(X_train, y_train)
# # Training Model : Predict
# y_pred = logreg.predict(X_test)
#
# # Evaluation Model
# print("Training Accuracy :", logreg.score(X_train, y_train))
# print("Testing Accuracy :", logreg.score(X_test, y_test))
#
# # Confusion Matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# print("\nConfusion Matrix :\n", cnf_matrix)
#
# # Plotting Confusion Matrix
# class_name = [0, 1]
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_name))
# plt.xticks(tick_marks, class_name)
# plt.yticks(tick_marks, class_name)
#
# # Create heatmap
# sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
# ax.xaxis.set_label_position('top')
# plt.title("Confusion Matrix", y=1.1)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.tight_layout()
# plt.show()
