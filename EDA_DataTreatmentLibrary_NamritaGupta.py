'''
This is a custom library which have a parent class "EDA_attributes" which gives the basic statistical summary.
Another class "Distribution_Treatment" inherit from this parent class.
Class "Distribution_Treatment" have three methods:
1.distribution_plot - This method gives the distribution of customers across different attributes gender, age, marital status, number of years in that city and city.
2.purchase_plot - This method gives the distribution of purchase pattern across different attributes gender, age, marital status, number of years in that city and city.
3.attribute_treatment - Categorical Attribute that have ordinal values (that can be arranged in order like Age) got Converted from categorical to numrical.
'''

#Importing 3 party library Matplotlib which would help in plotting of graphs used in performing exploratory data analysis (EDA)
import matplotlib.pyplot as plt

#Making a parent class EDA_attributes which would give the basic statistical summary
class EDA_attributes ():
    #Initialising the class and assigning the variable 'dataframe' so that dataframe from the main program can be passed
    def __init__(self, dataframe):
        self.dataframe = dataframe

    #Performing the basic stastics and priniting the results            
    def get_basic_stats(self):
        basic_stats = self.dataframe.describe()
        return(print("Basic Statistical Summary",'\n',basic_stats))

#Making a subclass that inherit method 'get_basic_stats' from parent class EDA_attributes
class Distribution_Treatment(EDA_attributes):
    #This method would plot the distribution of customers across various attributes for EDA
    #This method require two arguments attribute and title for the graph
    def distribution_plot(self,attribute,title):
        #For attributes : 'Gender','City_Category','Marital_Status' pie graph is plotted
        attri_list1 = ['Gender','City_Category','Marital_Status']
        #Using if for decision making that if the attributes are 'Gender','City_Category','Marital_Status' then plotting pie graph else bar graph
        if attribute in attri_list1:
            fig_1,axis_1 = plt.subplots()
            #Calculating the total customers count group by the selected attribute
            axis_1 = self.dataframe[attribute].value_counts().plot(kind = 'pie',autopct='%1.1f%%')
            axis_1.set_title(title)
            axis_1.set_ylabel('Number of Customers')
            axis_1.set_xlabel(attribute)
            #This would export the result and produce an output graph that would be stored in the current folder
            return (plt.savefig(title))
        #For attributes : 'Age','Stay_In_Current_City_Years' bar graph is plottef
        else:
            fig_1,axis_1 = plt.subplots()
            #Calculating the total customers count group by the selected attribute
            axis_1 = self.dataframe[attribute].value_counts().plot(kind = 'bar',rot = 0, width = 0.4)
            axis_1.set_title(title)
            axis_1.set_ylabel('Number of Customers')
            axis_1.set_xlabel(attribute)
            #This would export the result and produce an output graph that would be stored in the current folder
            return (plt.savefig(title))
        
    #This method would plot the purchase pattern of customers across various attributes for EDA
    #This method require two arguments attribute and title for the graph        
    def purchase_plot(self,attribute,title):
        #Calculating the mean sales group by the selected attribute
        var = self.dataframe.groupby(attribute).Total_Purchase.mean()
        graph = plt.figure()
        ax1 = graph.add_subplot(1,1,1)
        ax1.set_xlabel(attribute)
        ax1.set_ylabel('Average_Purchase')
        ax1.set_title(title)
        var.plot(kind='bar')
        #This would export the result and produce an output graph that would be stored in the current folder
        return (plt.savefig(title))
        
    #In this method Categorical Attribute that have ordinal values (that can be arranged in order like Age) will get Converted from categorical to numrical.
    #This method require four arguments: attribute, number of unique values, initial values in a list, new values in a list
    def attribute_treatment(self,attribute,num_of_values,values,new_values):
        #Iterating as many times as the number of unique values
        for i in range(num_of_values):
            #Inner loop to change each value from categorical to numerical
            for j in range(len(values)):
                self.dataframe[attribute].replace(to_replace=values[j], value=new_values[j], inplace=True)
        #retuning the success message for each attribute
        return(print("Categorical Attribute",attribute,"got Converted from categorical to numrical"))


        
        
