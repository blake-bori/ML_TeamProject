# ML_TeamProject
 in-vehicle coupon recommendation (with autoML)

# 1. Objective
Project Title : in-vehicle coupon recommendation
Objective : We infer and recommend which coupons the in-vehicle user is likely to use.
Expectation effectiveness : Expect customers’ response to recommendations made in an in-vehicle recommender system that provides coupons for local businesses.

# 2. Dataset
Title : in-vehicle coupon recommendation Data Set
URL : https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation

Description
- collected via a survey on Amazon Mechanical Turk. 
- The survey describes different driving scenarios including the destination, current time, weather, passenger, etc.
- And then ask the person whether he will use the coupon if he is the driver.
 
 
Data Inspection



Attribution Information

- 12684 samples
- 25 predictor variables (explain later)
- target(for classification) : Y{0.1}
  (whether the coupon is accepted)

* ‘accepted’ means that it will be used.


Data Visualization 

      
      
 
# 3. Process
3.1. classificationFunc(), clusteringFunc()
Find the best combination of scaler and parameter for classification and clustering algorithm learning. 
As a result of classification, print the accuracy, confusion matrix, and ROC curve as a result of classification.
As a result of clustering, print a two-dimensional scatter plot graph, silhouette coefficient, and purity using PCA.

Parameter
data : data frame.
scalers : list of scalers.
scaled_cols : list of columns that will be scaled.
models_params : dictionary of ML algorithm and parameters. Here, 
            Parameters are the range for finding the best parameters.
cv : K-fold’s K value


Example
df = pd.read_csv("vehicle_coupon.csv")
classificationData = preprocessing(df, False)
classification_scaled_cols = classificationData.columns
scaled_cols = classification_scaled_cols.drop("Y")
scalers = ["StandardScaler", "MinmaxScaler", "RobustScaler"]

models_params_classifier = 
{"DecisionTreeClassifier": 
   {"max_depth": [5, 10], "criterion": ["gini", "entropy"], "max_features": 
   [None, "sqrt", "log2", 3, 4, 5], "max_leaf_nodes": [5, 10], 
   "min_samples_leaf": [5, 10], "min_samples_split": [5, 10], "random_state": 
   [5]}, 
"LogisticRegression": {"C": [0.1, 0.5, 1], "max_iter": [500, 1000], "solver": 
   ["newton-cg", "lbfgs", "saga"], "random_state": [5]}, 
"RandomForestClassifier": {'criterion': ['gini', "entropy"], "max_depth": [5, 
   10], "max_features": [None, "sqrt", "log2", 3, 4, 5], "max_leaf_nodes": [5, 
   10], "max_samples": [5, 10, 20], "n_estimators": [5, 10, 20], "random_state": 
   [5]}
}

classificationFunc(data, scalers, scaled_cols, models_params_classifier, 
                   5)




3.2. autoML() <bestFindParam function>

Find the best parameter values for the model and dataset. The model parameters received as input are initial values, and the best parameters outside this range can be found.

Ouput  :  best result = [model object, model parameter(dictionary), score]


AutoML Step
1.	Parameter lists are applied to a model to obtain a parameter set with the best accuracy among input parameters. (this process is conducted to ensure that, when autoML function is performed, it does not fall into the local maximum but reaches the global maximum)

2.	Increase accuracy by increasing or lowering the value by a predetermined interval each parameter. (If the value of the parameter is a categorical value, it will move on to the next parameter)

3.	If the accuracy has improved compared to the accuracy before repeating at the end of each parameter, go back to step 2, repeat it again, and return the final parameter set and result if there is no difference.



Parameter
x_train : features of train data
y_train : target of train data (clustering receives None)
model : learning model algorithm
model_params : dictionary type = {‘parameter’ : param_value(list)}
score : criteria for finding the best parameter
       (classification : accuracy, clustering : silhouette)
cv : K-fold’s K value


Example
df = pd.read_csv("vehicle_coupon.csv")
classificationData = preprocessing(df, False)
X_train_scaled = classificationData.drop("Y")
y_train = classificationData[‘Y’]
model = ‘DecisionTreeClassifier’
models_params ={"max_depth": [5, 10], "criterion": ["gini", "entropy"], "max_features": [None, "sqrt", "log2", 3, 4, 5], "max_leaf_nodes": [5, 10], "min_samples_leaf": [5, 10], "min_samples_split": [5, 10], "random_state": [5]}


findBestParam(X_train_scaled, y_train, model, models_params)

 
# 4. Result
Result of classification

________________________________________
Decision Tree Classifier
 
![image](https://user-images.githubusercontent.com/76082792/142419714-78563a93-6537-42a6-83af-5fe8ac333c52.png)
![image](https://user-images.githubusercontent.com/76082792/142419728-d257c946-67da-41c8-9a16-eb7b4571489c.png) 
 
________________________________________
Logistic Regression
 
![image](https://user-images.githubusercontent.com/76082792/142419746-258e03cc-cfe4-4e1e-9e7f-97529208c055.png)
![image](https://user-images.githubusercontent.com/76082792/142419783-62a900be-a7fb-4584-ba77-5157d3647ca1.png)
 
  
________________________________________
Random Forest Classifier
 
![image](https://user-images.githubusercontent.com/76082792/142419793-dc00fbad-9ebb-42ea-8444-a85c797e6c7a.png)
![image](https://user-images.githubusercontent.com/76082792/142419800-0c3edcf9-5ec6-4cea-9705-b60f4e3fc107.png)
 

  
________________________________________
 
Result of clustering
 
![image](https://user-images.githubusercontent.com/76082792/142419828-c628b8ff-58b6-4d31-a6da-d2fcce08122d.png)
 

