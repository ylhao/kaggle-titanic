#coding: utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../titanic_data/train.csv')
test_df = pd.read_csv('../titanic_data/test.csv')

#train_df = pd.read_csv('../titanic_data/train.csv', sep=',', header=0)
#test_df = pd.read_csv('../titanic_data/test.csv', sep=',', header=0)

'''
以上两段是等价的，read_csv是pandas提供的一个常用的函数，用于读取csv格式的数据
其中sep和header是常用的两个参数，sep指定分隔符，默认使用“，”，如果没有列名，可
将header指定为None。
'''

print type(train_df)

'''
打印了数据的格式： <class 'pandas.core.frame.DataFrame'>
pandas有两个主要的数据结构Series和DataFrame
DataFrame是一个表格型的数据结构，既有行索引，也有列索引
'''

print train_df.columns.values

'''
打印列名（前提是有列名，否则会报错）
'''

print train_df.head()
#print train_df.head(3)

'''
预览数据，默认显示前5行,可以通过注释的方式指定预览的行数
'''

print train_df.tail()
#print train_df.tail(3)

'''
预览数据
'''

print train_df.info()

'''
获取DataFrame的基本信息
<class 'pandas.core.frame.DataFrame'> ：该数据集是一个DataFrame实例
RangeIndex: 891 entries, 0 to 890 ：共有891行数据，行号为0～980
Data columns (total 12 columns): ：共有12列数据
PassengerId    891 non-null int64 ：
    该列数据的列名为PassengerId，共有891行数据非空，0行数据缺失，数据类型为int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5) ：数据类型的汇总信息
memory usage: 83.6+ KB ：保存该数据集耗费的内存
'''

print train_df.describe()

'''
描述数据集常用统计量信息
count/mean/std/min/25%/50%/75%/max
'''

print train_df.describe(include=['O'])

'''
给出object这类非数值变量的非空值数、unique数、最大频数变量，最大频数
'''

print train_df[2:4]

'''
用python常用的切片语法提取数据
'''

print train_df['Sex'].head()
print train_df[['Name', 'Sex']].head()

'''
我们可以把DataFrame看成有一组共享索引的Series组成
0      male
1    female
2    female
3    female
4      male
Name: Sex, dtype: object

                                                Name     Sex
0                            Braund, Mr. Owen Harris    male
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female
2                             Heikkinen, Miss. Laina  female
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female
4                           Allen, Mr. William Henry    male

'''

print train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

'''
数据分析
'''

#####

'''
可视化分析略
'''

combine = [train_df, test_df]
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [tran_df, test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

'''
丢弃一些作用不大的特征（丢弃了“Ticket”和“Cabin”这两列数据）
'''

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print pd.crosstab(train_df['Title'], train_df['Sex'])

'''
构造新的特征，提取名字的姓氏，DataFrame中添加了一列，列名为Title，然后分析姓氏跟性别的关系
'''

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print train_df['Title'].head(50)

'''
在以上分析的基础上，按以上方式修改Title这一列的值
'''

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print train_df.info()
print train_df.head()

'''
进一步处理Title这一列的数据，若有缺失值填补为0。
'''

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print train_df.shape, test_df.shape
print train_df.head()

'''
Name这个特征可以考虑丢弃了，这里丢弃这一列特征
'''

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print train_df.head()

'''
将sex这一列数据处理为数值型
'''

guess_ages = np.zeros((2, 3))
print guess_ages

'''
分别代表，sex=0, Pclass=1时的年龄填充值
         sex=0, Pclass=2时的年龄填充值
         …
'''

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j] # 填充所有的空缺值

    dataset['Age'] = dataset['Age'].astype(int) #设置数据类型为int64

'''
age特征的缺失值处理，基本方式为填充
'''

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

'''
划分年龄段，添加了一列AgeBand，并且分析各个年龄段和存活率的关系，
分析过后，发现年龄段和存活率之间存在关系，接下来进一步处理
'''

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

'''
将年龄映射到各自对应的年龄段
'''

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print train_df.head()

'''
删掉AgeBand这个添加的特征
'''

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

'''
添加新特征，并分析新特征和存活率之间的关系
'''

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

'''
添加代表是否孤身一人的新特征，并分析新特征和存活率之间的关系
'''

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

print train_df.head()

'''
丢弃和IsAlone重合的3个特征
疑问：这里是否可以考虑不丢弃，不丢弃效果是否会更好？
'''

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

'''
通过Age和Class构造人工特征
'''

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

'''
Embarked有缺失值，用众数填补缺失值
'''

print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

'''
分析Embarked特征和存活率之间的关系
'''

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
print train_df.head()

'''
将Embarked特征进行转换，并且指定数据类型为int64
'''

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

'''
测试集缺失值填补
'''

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

'''
添加新的特征FareBand，也就是将Fare分段，并且分析各个FareBand和存活率之间的关系
'''

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
print train_df.head(10)
print test_df.head(10)
print train_df.info()
print '-'*40
print test_df.info()
'''
Fare特征转换，丢弃掉FareBand特征，到此数据预处理结束
'''

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"] #标签
X_test  = test_df.drop("PassengerId", axis=1).copy()
print X_train.shape, Y_train.shape, X_test.shape

'''
构造训练集，测试集
'''

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print acc_log

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print acc_svc

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print acc_knn

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print acc_linear_svc

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print acc_random_forest

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest',  'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_linear_svc, acc_decision_tree]})
print models.sort_values(by='Score', ascending=False)

'''
各种方法的结果对比
'''

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
print submission.head()
submission.to_csv('../titanic_data/submission.csv', index=False)

'''
获得最后要提交的结果
'''