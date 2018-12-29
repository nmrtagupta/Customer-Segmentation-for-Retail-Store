'''
Author - Namrita Gupta
Final Project
---------------------------------------------------------------Customer Segmentation for Retail Store-----------------------------------------------------------------
Obejctive of the project is to come up with an approach that can be used to segment the customer based on their purchase pattern
General Methodolgy involved:
data import --> data cleaning --> data aggregation --> Exploratory data analysis (EDA) --> Data preperation for modelling --> Elbow method to know the optimal cluster numbers --> Clustering using K-means  
'''

#Custom Library "EDA_DataTreatmentLibrary" was created which was used in performing EDA and Data preperation for modelling
#Sub class "Distribution_Treatment" which inherited parent class "EDA_attributes"
#This was used for EDA and Data preperation for modelling
from EDA_DataTreatmentLibrary_NamritaGupta import Distribution_Treatment
#Parent class "EDA_attributes" which gives the basic statistical summary
from EDA_DataTreatmentLibrary_NamritaGupta import EDA_attributes

#Importing 3rd Party libraries
#Pandas is used for importing and data processing
import pandas as pd
#Matplotlib is used for plotting the EDA graphs
import matplotlib.pyplot as plt # 

#sklearn.cluster is used to perform K-means modelling technique which would perform clustering of the customers
from sklearn.cluster import KMeans

#----------------------------------------------------------------Data Import-----------------------------------------------------------------------------------------------
#Importing the raw data csv file called "RetailData" using pandas library which is stored as dataframe data structure
customer_data = pd.read_csv("RetailData.csv")

#Viewing 10 rows of the data using head
head = customer_data.head(10)
print ('Imported data head','\n',head)

#----------------------------------------------------------------Data Cleaning and Aggregation-----------------------------------------------------------------------------------------------
#The current data represents the transactional detail of the customers. The data is at transaction level which is converted to customer level.

#Aggregating the totol products that is bought by the customer
tot_item = customer_data['User_ID'].value_counts().sort_index()
#Aggregating the totol purchase that is done by the customer
tot_purchase = customer_data.groupby('User_ID').sum()['Purchase']
#Concatenating the aggregated columns together
aggregated_attribures = pd.concat([tot_item, tot_purchase], axis = 1, keys = ['Total_Products', 'Total_Purchase'])

#Merging of the concatenated columns and imported data
customer_data = pd.merge(customer_data, aggregated_attribures, left_on = 'User_ID', right_index = True)

#Analyzing which attributes have missing values
MissingValues = customer_data.isnull().sum().sort_values(ascending = False)
print('\n',"MissingValues",'\n',MissingValues)

#Removing the attributes with missing values 'Product_Category_2', and 'Product_Category_3'
#Also removing attribute 'Product_ID','Product_Category_1','Purchase' as they are already aggregated into new columns
customer_data.drop(['Product_ID', 'Product_Category_1','Product_Category_2','Product_Category_3','Purchase'], axis = 1, inplace = True)

#Total product and total purchase are aggregated and merged back to data and the above columns are removed
#Now the data is in this format, the attributes are aggregated but it is still have duplicate rows
head = customer_data.head(10)
print ('\n','Data before roll up','\n',head)

#Rolling up the data by deleting the duplicates
customer_data.drop_duplicates(inplace = True)

#Final Customer level data
head = customer_data.head(10)
print ('\n','Final Customer level data','\n',head)

#----------------------------------------------------------------Exploratory Data Analysis-----------------------------------------------------------------------------------------------
# Exploratory data analysis - Univariate Analysis of the data
#All the graphs are exported to the current folder

#Making the object univariate using Distribution_Treatment class to perform EDA  
univariate = Distribution_Treatment(customer_data)

#distribution_plot - This method gives the distribution of customers across different attributes gender, age, marital status, number of years in that city and city.
#Pie Chart for Distibution of customers by Gender
univariate.distribution_plot('Gender','Distibution of customers by Gender')

#Pie Chart for Distibution of customers across Cities
univariate.distribution_plot('City_Category','Distibution of customers across city')

#Pie Chart for Distibution of customers by Marital Status
univariate.distribution_plot('Marital_Status','Distibution of customers by marital status')

#Bar Graph for Distibution of customers across age buckets
univariate.distribution_plot('Age','Distibution of customers across Age buckets')

#Bar Graph for Distibution of customers across tenure of stay in current city in years
univariate.distribution_plot('Stay_In_Current_City_Years','Distibution of customers across tenure of stay in current city in years')

#purchase_plot - This method gives the distribution of purchase pattern across different attributes
#gender, age, marital status, number of years in that city and city.
#Purchase pattern by Gender attribute
univariate.purchase_plot('Gender','Puchase Pattern by Gender')

#Purchase pattern across City_Category attribute
univariate.purchase_plot('City_Category','Puchase Pattern across City_Category')

#Purchase pattern by Marital_Status attribute
univariate.purchase_plot('Marital_Status','Puchase Pattern by Marital_Status')

#Purchase pattern across Age attribute
univariate.purchase_plot('Age','Puchase Pattern across Age')

#Purchase pattern across Stay_In_Current_City_Years attribute
univariate.purchase_plot('Stay_In_Current_City_Years','Puchase Pattern across tenure of stay in current city in years')

#Getting the basic statistical summary using the inheritance property by calling the method of parent class
#Statistical_summary = Distribution_Treatment(customer_data)
univariate.get_basic_stats()

#----------------------------------------------------------------Data preperation for modelling-----------------------------------------------------------------------------------------------

#For executing K-means algorithm all the attributes should be numerical as it assigns cluster based on the Euclidean distance

#Using the third method of the class "Distribution_Treatment"
#attribute_treatment - Categorical Attribute that have ordinal values (that can be arranged in order like Age) got Converted from categorical to numrical.

#Converting gender from categorical to numerical
values_gender = ['F','M']
new_values_gender = [1,0]
univariate.attribute_treatment('Gender',2,values_gender,new_values_gender)

#Converting age from categorical to numerical
values_age = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
new_values_age = [1,2,3,4,5,6,7]
univariate.attribute_treatment('Age',7,values_age,new_values_age)

#Converting Stay_In_Current_City_Years from categorical to numerical
value_stay = ['0','1','2','3','4+']
new_values_stay = [1,2,3,4,5]
univariate.attribute_treatment('Stay_In_Current_City_Years',5,value_stay,new_values_stay)

'''
Attribute like City_Category don't have order in the values like the difference between City A and City B is not same as the difference between City A and City C
which was the case with other attributes like age.Such attributes would be converted from text to column with binary values in it. Like City A, City B and City C
would be three column with binnary value indicating whether the customer belong to that city or not
'''
customer_data=pd.get_dummies(data=customer_data)

#Exporting the final prepared dataset so that it can be directly used for applying other modeling technique 
customer_data.to_csv('Final_Prepared_Dataset.csv', sep=',')

#----------------------------------------------------------------Elbow method to know the optimal cluster numbers-----------------------------------------------------------------------------------------------
'''
Elbow method is used to used to get the optimal number of clusters that have to be given in the input of K-means algorithm.
The optimal number of clusters are chosed from the elbow point in the graph 
'''

#Creatin a list to store the sum of squared error on each number
sum_squared_error = []
#Running 10 iteration to get the best fit
for i in range(1, 11):
    #using KMeans method from sklearn.cluster
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(customer_data)
    sum_squared_error.append(kmeans.inertia_)
#Creating graph for the elbow plot
plt.plot(range(1, 11), sum_squared_error)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared error')
#The elbow plot is exported to the current folder
plt.savefig('Elbow Plot')

#----------------------------------------------------------------Clustering of customers using K-means -----------------------------------------------------------------------------------------------

#Segementing of customers by using K-means algorithm from sklearn.cluster

#Executing the Kmeans algorithm with optimal cluster number from elbow plot
Kmeans_result = KMeans(3)

#Fitting the data to the model
Kmeans_result.fit(customer_data)

#Predicting the cluster labels
cluster_labels = Kmeans_result.predict(customer_data)

#Coverting the cluster_labels to the dataframe structure using pandas
kmeans_cluster = pd.DataFrame(cluster_labels)

#Merging the clusters to the dataset
customer_data.insert((customer_data.shape[1]),'cluster',kmeans_cluster)

#Visualising the clusters using matplotlib
kmeans_fig = plt.figure()
kmeans_graph = kmeans_fig.add_subplot(111)
scatter = kmeans_graph.scatter(customer_data['Total_Products'],customer_data['Total_Purchase'],
                     c=kmeans_cluster[0],s=50)
kmeans_graph.set_title('K-Means Clustering')
kmeans_graph.set_xlabel('Total_Products')
kmeans_graph.set_ylabel('Total_Purchase')
#Exporting the clustering result to the current folder by name "K-means Cluster Result"
plt.savefig("K-means Cluster Result")

#----------------------------------------------------------------Thankyou -----------------------------------------------------------------------------------------------








