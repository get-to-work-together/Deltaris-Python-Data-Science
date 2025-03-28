#!/usr/bin/env python
# coding: utf-8

# # Classification with Penguin Dataset

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the data

# In[2]:


df = sns.load_dataset('penguins')
df.head()


# ## Exploratory Data Analysis (EDA)

# In[3]:


df.info()


# In[4]:


sns.pairplot(df, hue='species');


# ## Remove rows with missing data

# In[5]:


df.dropna(inplace = True)


# In[6]:


df.info()


# ## Identify target and feature columns

# In[7]:


target = 'species'

numeric_features = [
    'bill_length_mm', 
    'bill_depth_mm',
    'flipper_length_mm', 
    'body_mass_g', 
]

categorical_features = [
    'island', 
    'sex'
]

features = numeric_features + categorical_features


# ## Target

# In[8]:


df[target].value_counts()


# In[9]:


df[target] = df[target].astype('category')


# In[10]:


df_target = df[target]


# ## One hot encoding of categorical features

# In[11]:


df_dummy_variables = pd.get_dummies(df[categorical_features], dtype='int', drop_first=True)
df_dummy_variables


# In[12]:


df_features = pd.concat([df[numeric_features], df_dummy_variables], axis=1)
df_features


# ## Split in training and test dataset

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


df_target_train, df_target_test, df_features_train, df_features_test = \
    train_test_split(df_target, df_features, test_size = 0.3, random_state = 12345)

print('df_target_train', df_target_train.shape)
print('df_target_test', df_target_test.shape)
print('df_features_train', df_features_train.shape)
print('df_features_test', df_features_test.shape)


# ## Scale numeric features

# In[15]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[16]:


scaler = MinMaxScaler()

scaler.fit(df_features_train)

train_features_scaled = scaler.transform(df_features_train)
test_features_scaled = scaler.transform(df_features_test)

print('mean', round(train_features_scaled.mean(), 10))
print('std', train_features_scaled.std())
print('min', train_features_scaled.min())
print('max', train_features_scaled.max())


# In[17]:


df_features_train_scaled = pd.DataFrame(train_features_scaled, 
                                        columns = [col + '_scaled' for col in list(df_features_train.columns)], 
                                        index = df_features_train.index)


# In[19]:


df_features_test_scaled = pd.DataFrame(test_features_scaled, 
                                       columns = [col + '_scaled' for col in list(df_features_test.columns)], 
                                       index = df_features_test.index)


# # Classification

# ## k-Nearest Neighbor (k=3)

# In[20]:


from sklearn.neighbors import KNeighborsClassifier


# In[21]:


classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(df_features_train_scaled, df_target_train)


# In[38]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)
print(f'{n_correct} of {n_total} correct')


# In[39]:


accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## k-Nearest Neighbor (k=5)

# In[40]:


classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(df_features_train_scaled, df_target_train)


# In[41]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)
print(f'{n_correct} of {n_total} correct')


# In[42]:


accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## k-Nearest Neighbor (k=5) without scaling

# In[47]:


classifier = KNeighborsClassifier(n_neighbors = 5)

classifier.fit(df_features_train, df_target_train)


# In[50]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)
print(f'{n_correct} of {n_total} correct')


# In[51]:


accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## Naive Bayes

# In[53]:


from sklearn.naive_bayes import GaussianNB


# In[54]:


classifier = GaussianNB()

classifier.fit(df_features_train_scaled, df_target_train)


# In[56]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## Decision Tree

# In[57]:


from sklearn.tree import DecisionTreeClassifier


# In[68]:


classifier = DecisionTreeClassifier(max_depth=3)

classifier.fit(df_features_train, df_target_train)


# In[69]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# In[82]:


from sklearn.tree import plot_tree

plt.figure(figsize = (15,10))

plot_tree(classifier, 
          feature_names = df_features_train.columns,
          class_names = df_target_train.unique(),
          filled = True,
          rounded = True, 
          impurity = False)

plt.show()


# In[78]:


from sklearn.tree import export_text

print(export_text(classifier, 
                  feature_names = df_features_train.columns))


# In[96]:


print('Feature Importances Decision Tree', classifier.feature_importances_)


# In[97]:


print('Feature Importances Decision Tree')
for k, v in zip( df_features_train.columns, classifier.feature_importances_):
    print(f'{k:20}: {v:.2f}')


# ## Random Forest

# In[98]:


from sklearn.ensemble import RandomForestClassifier


# In[99]:


classifier = RandomForestClassifier(n_estimators = 100)

classifier.fit(df_features_train, df_target_train)


# In[100]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# In[101]:


print('Feature Importances Decision Tree')
for k, v in zip( df_features_train.columns, classifier.feature_importances_):
    print(f'{k:20}: {v:.2f}')


# ## Logistic Regression

# In[108]:


from sklearn.linear_model import LogisticRegression


# In[109]:


classifier = LogisticRegression(random_state=0, max_iter=1000)

classifier.fit(df_features_train_scaled, df_target_train)


# In[110]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# In[114]:


classifier.coef_.round(2)


# ## Support Vector Machine

# In[103]:


from sklearn.svm import SVC


# In[104]:


classifier = SVC(kernel = 'rbf')

classifier.fit(df_features_train_scaled, df_target_train)


# In[106]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## Neural Network - Multi Layer Perceptron

# In[115]:


from sklearn.neural_network import MLPClassifier


# In[116]:


classifier = MLPClassifier(solver='lbfgs',
                               activation='relu',     #'logistic',
                               alpha=1e-5,
                               hidden_layer_sizes=(5, 5), 
                               random_state=1,
                               max_iter=10000)

classifier.fit(df_features_train_scaled, df_target_train)


# In[117]:


actual = df_target_test.values
predicted = classifier.predict(df_features_test_scaled)

correct = predicted == actual

n_correct = np.sum(correct)
n_total = len(actual)

accuracy = n_correct / n_total

print(f'Accuracy: {accuracy}')


# ## Performance metrics

# In[118]:


prediction = '5NN_predicted'


# ### confusion matrix

# In[119]:


from sklearn.metrics import confusion_matrix


# In[120]:


cm = confusion_matrix(df_target_test, predicted)
cm


# ### accuracy

# In[122]:


from sklearn.metrics import accuracy_score


# In[124]:


accuracy = accuracy_score(df_target_test, predicted)

print('accuracy:', accuracy)


# ### precision, recall and f-score

# In[126]:


from sklearn.metrics import precision_recall_fscore_support


# In[153]:


precision, recall, f_score, _ = precision_recall_fscore_support(df_target_test, predicted)

print('precision:', precision)
print('recall:', recall)
print('f-score:', f_score)


# ### all together

# In[183]:


def print_metrics(name, target, predicted, labels = ('Adelie', 'Gentoo', 'Chinstrap')):
    conf = confusion_matrix(target, predicted)
    accuracy = accuracy_score(target, predicted)
    precision, recall, f_score, num_case = precision_recall_fscore_support(target, predicted)
    
    print(80 * '=')
    print()
    print(name)
    print()
    print('Confusion matrix')
    print()
    print('                %20s %20s %20s' % tuple('Predicted ' + label for label in labels))
    print('Actual %-12s           %6d               %6d               %6d' % (labels[0:1] + tuple(conf[0])))
    print('Actual %-12s           %6d               %6d               %6d' % (labels[1:2] + tuple(conf[1])))
    print('Actual %-12s           %6d               %6d               %6d' % (labels[2:3] + tuple(conf[2])))
    print()
    print('Accuracy  %0.2f' % accuracy)
    print()
    print('                %20s %20s %20s' % tuple(labels))
    print('Number of cases %20d %20d %20d' % tuple(num_case))
    print('Precision       %20.2f %20.2f %20.2f' % tuple(precision))
    print('Recall          %20.2f %20.2f %20.2f' % tuple(recall))
    print('F1              %20.2f %20.2f %20.2f' % tuple(f_score))
    print()


# In[184]:


def classify(name, classifier, df_features_train, df_target_train, df_features_test, df_target_test):
    classifier.fit(df_features_train, df_target_train)

    actual = df_target_test.values
    predicted = classifier.predict(df_features_test)
    
    print_metrics(name, actual, predicted)


# In[185]:


classifiers = {
    '3 Nearest Neighbor': KNeighborsClassifier(n_neighbors = 3),
    '5 Nearest Neighbor': KNeighborsClassifier(n_neighbors = 5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=3),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
    'Support Vector Machine': SVC(kernel = 'rbf'),
    'Multi Layer Perceptron': MLPClassifier(hidden_layer_sizes=(5, 5), activation='relu', solver='lbfgs', max_iter=10000)
}

for name, classifier in classifiers.items():
    classify(name, 
             classifier, 
             df_features_train_scaled, 
             df_target_train, 
             df_features_test_scaled, 
             df_target_test)


# In[ ]:





# In[ ]:





# In[ ]:




