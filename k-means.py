import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
 
def elbow(scaled_df, title):
    wcss=[]
    for i in range(1,20):
        km = KMeans(i)
        km.fit(scaled_df)
        wcss.append(km.inertia_)
    np.array(wcss)
    print('__________________________________________________')
    print(title)
    return wcss
 
def graphics_for_df_and_new_df(title, wcss, elbow_point, elbow_point_no):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = plt.plot(range(1,20), wcss, linewidth=2, color="red", marker ="8")
    ax_no = plt.plot(range(1,20), wcss_no, linewidth=2, color="blue", marker ="8")
    plt.axvline(x=elbow_point, ls='-', color="red")
    plt.axvline(x=elbow_point_no, ls=':', color="blue")
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.xlabel('Number of Clusters')
    plt.xticks(np.arange(0, 20, 1)) #generates an array of numbers from 0 to 9 with a step size of 1
    plt.title(title, fontsize = 18)
    plt.show()
 
def silhouette(X, title):
    silhouette_average = []
    for k in range(2,20):
        kmeans = KMeans(n_clusters = k).fit(X)
        labels = kmeans.labels_
        silhouette_average.append(silhouette_score(X, labels, metric = 'euclidean'))
    optimal_n_clusters = range(2,20)[np.argmax(silhouette_average)]
    print('Optimal K depending on the silhouette score: ' + str(optimal_n_clusters))
    plt.plot(range(2,20),silhouette_average,'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title(title)
    plt.show()

def visualization(new_df, title, km_no):
    plt.figure(figsize=(10,7))
    plt.scatter(new_df['Annual Income (k$)'], new_df['Spending Score (1-100)'],  c=km_no.labels_ , cmap='rainbow')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)') 
    plt.title(title)
    plt.show()
 
data = 'Mall_Customers.csv' #import a csv file with the data we are going to use
df = pd.read_csv(data) #read from 'Mall_Customers.csv'
 
#FILE PROCESSING
encoder = OneHotEncoder(handle_unknown='ignore') #create instance of one-hot-encoder
encoder_df = pd.DataFrame(encoder.fit_transform(df[['Gender']]).toarray()) #perform one-hot encoding on 'Gender' column
df = df.join(encoder_df) #merge one-hot encoded columns back with original DataFrame
df = df.drop('CustomerID', axis=1) #drop 'CustomerID' column
df.drop('Gender', axis=1, inplace=True) #drop 'Gender' column -> 0 - male, 1 - female
df.rename(columns={0: 'isMale', 1: 'isFemale'}, inplace=True) #replace 0 to 'isMale' and 1 to 'isFemale'
df.to_csv('Hot_Encoded.csv', index=False) #save in "Hot_Encoded.csv"
df.to_csv('Hot_Encoded_No_Outliers.csv', index=False) #save to 'Hot_Encoded_No_Outliers.csv' as well
new_data = 'Hot_Encoded_No_Outliers.csv' #save the file 'Hot_Encoded_No_Outliers.csv' in a variable called 'new_data'
new_df = pd.read_csv(new_data) #read from 'Hot_Encoded_No_Outliers.csv'
 
#OUTLIER PROCESSING
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in columns:
    shapiro_test = shapiro(new_df[col])
    print('__________________________________________________')
    print('Pvalue of column ' + col + ': ' + str(shapiro_test.pvalue))
    plt.boxplot(df[col])
    plt.title(col, fontsize = 20)
    plt.show() #show outliers for every column
    if shapiro_test.pvalue < 0.05: #find outliers from a data that is normally distributed 
        print('The data from ' + col + ' is not normally distributed,' +  
        ' that\'s why we should use the IQR (Interquartile Range) method for identifying the outliers.')
        Q1 = new_df[col].quantile(0.25)
        Q3 = new_df[col].quantile(0.75)
        IQR = Q3 - Q1
        print('IQR of ' + col + ': ' + str(IQR))
        for i in range(new_df[col].size):
            value = new_df[col][i]
            if value < (Q1 - 1.5 * IQR) or value > (Q3 + 1.5 * IQR): #pattern for checking for outliers
                print('Outlier value: ' + str(value) + ' in ' + col)
                new_df.loc[i, col] = 'Outlier'
                new_df.to_csv(new_data, index=False)
 
    else: #find outliers from a data that is not normally distributed
        print('The data from ' + col + ' is normally distributed,' +  
        ' that\'s why we should use the Z-score method for identifying the outliers.')
        mean = np.mean(col)
        std = np.std(col)
        print('The mean of the dataset is ', mean)
        print('The standart deviation is ', std)
        for i in range(new_df[col].size):
            value = new_df[col][i]
            z = (value - mean) / std
            if z > 3: #pattern for checking for outliers
                print('Outlier value: ' + str(value) + ' in ' + col)
                new_df.loc[i, col] = 'Outlier'
                new_df.to_csv(new_data, index=False)
for col in columns:
    for row in new_df[col]:
        if row == 'Outlier':
            new_df.drop(new_df.index[new_df[col] == row],inplace=True) #remove the row that has an outlier
            new_df.to_csv(new_data, index=False) #save to file   
 
#NORMALIZE DATA WITH OUTLIERS
df = pd.read_csv('Hot_Encoded.csv')
d = preprocessing.normalize(df)
names = df.columns
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.to_csv('Hot_Encoded.csv', index=False)
 
#NORMALIZE DATA WITHOUT OUTLIERS
new_df = pd.read_csv('Hot_Encoded_No_Outliers.csv')
copy_of_hot_encoded = new_df.to_csv('Hot_Encoded_No_Out_Not_Norm.csv', index=False)
new_df = pd.read_csv('Hot_Encoded_No_Outliers.csv')
d = preprocessing.normalize(new_df)
names = new_df.columns
scaled_new_df = pd.DataFrame(d, columns=names)
scaled_new_df.to_csv('Hot_Encoded_No_Outliers.csv', index=False)

#ELBOW METHOD WITH OUTLIERS
wcss = elbow(scaled_df, 'Array of predicted clusters in which every element belongs for all the columns with outliers:')
kn = KneeLocator(range(1,20), wcss, curve='convex', direction='decreasing')
elbow_point = kn.knee
km = KMeans(n_clusters = elbow_point)
y_predicted = km.fit_predict(df[['Age','Annual Income (k$)', 'Spending Score (1-100)', 'isMale', 'isFemale']])
print(y_predicted)
 
#ELBOW METHOD WITHOUT OUTLIERS
wcss_no = elbow(scaled_new_df, 'Array of predicted clusters in which every element belongs for all the columns without outliers:')
kn_no = KneeLocator(range(1,20), wcss_no, curve='convex', direction='decreasing')
elbow_point_no = kn_no.knee
km_no = KMeans(n_clusters = elbow_point_no)
y_predicted_no = km_no.fit_predict(new_df[['Age','Annual Income (k$)', 'Spending Score (1-100)', 'isMale', 'isFemale']])
print(y_predicted_no)
 
#GRAPHICS FOR WCSS AND WCSS_NO
graphics_for_df_and_new_df('The Elbow Method for the data with(red) and without(blue) the outliers', wcss, elbow_point, elbow_point_no)
 
#SILHOUETTE METHOD WITH OUTLIERS
print('__________________________________________________')
print('Silhouette method WITH OUTLIERS:')
silhouette(df[['Annual Income (k$)', 'Spending Score (1-100)']], 'Silhouette method WITH OUTLIERS')
 
#SILHOUETTE METHOD WITHOUT OUTLIERS
print('Silhouette method WITHOUT OUTLIERS:')
silhouette(new_df[['Annual Income (k$)', 'Spending Score (1-100)']], 'Silhouette method WITHOUT OUTLIERS')
 
#elbow 2 for two columns only WITH OUTLIERS
wcss = elbow(scaled_df, 'Array of predicted clusters in which every element belongs for two columns only with outliers:')
kn = KneeLocator(range(1,20), wcss, curve='convex', direction='decreasing')
elbow_point = kn.knee
 
km = KMeans(n_clusters = elbow_point)
y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
print(y_predicted)
 
#WITHOUT OUTLIERS
wcss_no = elbow(scaled_new_df, 'Array of predicted clusters in which every element belongs for two columns only without outliers:')
kn_no = KneeLocator(range(1,20), wcss_no, curve='convex', direction='decreasing')
elbow_point_no = kn_no.knee
 
km_no = KMeans(n_clusters = elbow_point_no)
y_predicted_no = km_no.fit_predict(new_df[['Annual Income (k$)', 'Spending Score (1-100)']])
print(y_predicted_no)
 
#GRAPHICS FOR WCSS AND WCSS_NO
graphics_for_df_and_new_df('Data with(red) and without(blue) the outliers FOR TWO COLUMNS ONLY', wcss, elbow_point, elbow_point_no)
 
#Scatterplot of the input data
visualization(df, 'Spending Score (1-100) vs Annual Income (k$) WITH OUTLIERS', km)

visualization(new_df, 'Spending Score (1-100) vs Annual Income (k$) WITHOUT OUTLIERS', km_no)

#ANALYSIS
mall_df = pd.read_csv('Hot_Encoded_No_Out_Not_Norm.csv') #read 'Hot_Encoded_No_Out_Not_Norm.csv'
mall_df['Cluster_Num'] = y_predicted_no # add a new column to the file that has the data for each cluster
mall_df.to_csv('Hot_Encoded_No_Out_Not_Norm.csv', index=False)

clusters=mall_df[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster_Num']]
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15,20))
ax[0,0].scatter(x=clusters[clusters['Cluster_Num'] == 0]['Annual Income (k$)'],
                y=clusters[clusters['Cluster_Num'] == 0]['Spending Score (1-100)'],
                s=40,edgecolor='black', linewidth=0.3, c='orange', label='Cluster 0')
ax[0,0].scatter(x=km_no.cluster_centers_[0,0], y=km_no.cluster_centers_[0,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[0,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 0')

ax[0,1].scatter(x=clusters[clusters['Cluster_Num'] == 1]['Annual Income (k$)'],
            y=clusters[clusters['Cluster_Num'] == 1]['Spending Score (1-100)'],
            s=40,edgecolor='black', linewidth=0.3, c='deepskyblue', label='Cluster 1')
ax[0,1].scatter(x=km_no.cluster_centers_[1,0], y=km_no.cluster_centers_[1,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[0,1].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 1')

ax[1,0].scatter(x=clusters[clusters['Cluster_Num'] == 2]['Annual Income (k$)'],
            y=clusters[clusters['Cluster_Num'] == 2]['Spending Score (1-100)'],
            s=40,edgecolor='black', linewidth=0.2, c='Magenta', label='Cluster 2')
ax[1,0].scatter(x=km_no.cluster_centers_[2,0], y=km_no.cluster_centers_[2,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[1,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 2')

ax[1,1].scatter(x=clusters[clusters['Cluster_Num'] == 3]['Annual Income (k$)'],
            y=clusters[clusters['Cluster_Num'] == 3]['Spending Score (1-100)'],
            s=40,edgecolor='black', linewidth=0.3, c='red', label='Cluster 3')
ax[1,1].scatter(x=km_no.cluster_centers_[3,0], y=km_no.cluster_centers_[3,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3)
ax[1,1].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 3')

ax[2,0].scatter(x=clusters[clusters['Cluster_Num'] == 4]['Annual Income (k$)'],
            y=clusters[clusters['Cluster_Num'] == 4]['Spending Score (1-100)'],
            s=40,edgecolor='black', linewidth=0.3, c='lime', label='Cluster 4')
ax[2,0].scatter(x=km_no.cluster_centers_[4,0], y=km_no.cluster_centers_[4,1],
                s = 120, c = 'yellow',edgecolor='black', linewidth=0.3, label='Centroids')
ax[2,0].set(xlim=(0,140), ylim=(0,100), xlabel='Annual Income', ylabel='Spending Score', title='Cluster 4')

fig.delaxes(ax[2,1])
fig.legend(loc='right')
fig.suptitle('Individual Clusters')
plt.show()

print('__________________________________________________')
cluster_0_df = mall_df[mall_df['Cluster_Num'] == 0]
cluster_1_df = mall_df[mall_df['Cluster_Num'] == 1]
cluster_2_df = mall_df[mall_df['Cluster_Num'] == 2]
cluster_3_df = mall_df[mall_df['Cluster_Num'] == 3]
cluster_4_df = mall_df[mall_df['Cluster_Num'] == 4]

left_coordinates=[0,1,2,3,4]
heights=[cluster_0_df['Annual Income (k$)'].mean(),cluster_1_df['Annual Income (k$)'].mean(),cluster_2_df['Annual Income (k$)'].mean(),cluster_3_df['Annual Income (k$)'].mean(),cluster_4_df['Annual Income (k$)'].mean()]
bar_labels=['Zero','One','Two','Three','Four']
plt.bar(left_coordinates,heights,tick_label=bar_labels,width=0.6,color='blue')
plt.xlabel('Cluster Number')
plt.ylabel('Annual Income mean')
plt.title('Annual Income mean for each cluster')
plt.show()

left_coordinates=[0,1,2,3,4]
heights=[cluster_0_df['Spending Score (1-100)'].mean(),cluster_1_df['Spending Score (1-100)'].mean(),cluster_2_df['Spending Score (1-100)'].mean(),cluster_3_df['Spending Score (1-100)'].mean(),cluster_4_df['Spending Score (1-100)'].mean()]
bar_labels=['Zero','One','Two','Three','Four']
plt.bar(left_coordinates,heights,tick_label=bar_labels,width=0.6,color='pink')
plt.xlabel('Cluster Number')
plt.ylabel('Spending Score mean')
plt.title('Spending Score mean for each cluster')
plt.show()

left_coordinates=[0,1,2,3,4]
heights=[cluster_0_df['Age'].mean(),cluster_1_df['Age'].mean(),cluster_2_df['Age'].mean(),cluster_3_df['Age'].mean(),cluster_4_df['Age'].mean()]
bar_labels=['Zero','One','Two','Three','Four']
plt.bar(left_coordinates,heights,tick_label=bar_labels,width=0.6,color='red')
plt.xlabel('Cluster Number')
plt.ylabel('Age Mean')
plt.title('Age mean for each cluster')
plt.show()

#GENDER ANALYSIS
X = ['Cluster Zero','Cluster One','Cluster Two','Cluster Three', 'Cluster Four']
Ygirls = [cluster_0_df['isFemale'].sum(),cluster_1_df['isFemale'].sum(),cluster_2_df['isFemale'].sum(),cluster_3_df['isFemale'].sum(),cluster_4_df['isFemale'].sum()]
Zboys = [cluster_0_df['isMale'].sum(),cluster_1_df['isMale'].sum(),cluster_2_df['isMale'].sum(),cluster_3_df['isMale'].sum(),cluster_4_df['isMale'].sum()]
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Ygirls, 0.4, label = 'Women')
plt.bar(X_axis + 0.2, Zboys, 0.4, label = 'Men')
  
plt.xticks(X_axis, X)
plt.xlabel('Cluster Number')
plt.ylabel('Gender Difference')
plt.title('Gender Difference for each cluster')
plt.legend()
plt.show()