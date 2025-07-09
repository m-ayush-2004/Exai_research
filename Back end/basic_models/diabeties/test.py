# tools for data explanation and visualization
import shap
import seaborn as sns
from sklearn.metrics import confusion_matrix
# tools for data manipulation and munging
import pandas as pd
import numpy as np
# tools that create regression models 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# we can use this to one hot encode all the values of tables in pandas from string to lable int
for i in x.columns:
    if(x[i].dtype== 'O'):
        print(f'replacing: {x[i].unique()}, to: {[i for i in range( len(x[i].unique()))]}')
        x.replace(x[i].unique(),[i for i in range( len(x[i].unique()))],inplace=True)
# here we are running the SHAP analysis algorithm that analyses the model based on  
# the concepts of game theory and can be used to explain the predictions of any machine learning model by calculating the contribution of each feature to the prediction.
explainer = shap.Explainer(model.predict, xtest)
shap_values = explainer(xtest)

# this is a plot of the datapoints of shap model that explains the role of each feature on y-axis
# and its affects on the model's predicted value where red is high feature value and blue is low feature value that negatively or positively affects the prediction of the model
shap.summary_plot(shap_values,plot_type='violin')

# generating a visual analysis of predictions using confusion matrix and seaborn library to generate matrix and heatmaps respectively
res = confusion_matrix(pred,ytest)
sns.heatmap(res, annot=True, fmt='g')
make it dynamic using plotly
sns.pairplot(tips,hue=target,palette='coolwarm')
hue wrt the target column pallete cool warm
x = titanic.corr(numeric_only=True)
sns.heatmap(x, cmap='coolwarm')
plt.title('correlation plot')