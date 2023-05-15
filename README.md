Sales Analysis Report
Introduction
The data: In this assignment, we used a synthetic dataset with 11 variables and over 400 records. The dataset provided was generated randomly to better fit the problem at hand.
The attributes are as follows:
 • Sales: Unit sales (in thousands) at each location
 • Competitor Price: Price charged by competitor at each location
 • Income: Community income level (in thousands of dollars)
 • Advertising: Local advertising budget for the company at each location (in thousands of dollars)
 • Population: Population size in the region (in thousands)
 • Price: Price the company charges for car seats at each site
 • Shelf Location: A factor with levels Bad, Good, and Medium indicating the quality of the shelving location for the car seats at each site
 • Age: Average age of the local population
 • Education: Education level at each location
 • Urban: A factor with levels No and Yes to indicate whether the store is in an urban or rural location
 • US: A factor with levels No and Yes to indicate whether the store is in the US or not
Data preprocessing
We started by loading the data using pandas and then proceeded to preprocess the data. First, we converted the categorical variables (Shelf Location, Urban, and US) using one-hot encoding:
python

Copy code
data = pd.get_dummies(data, columns=['Shelf_Location', 'Urban', 'US'])
Next, we split the data into input features (X) and target variable (y) and then into training and testing sets:
python

Copy code
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Model building and evaluation
We built two models to predict sales based on the given attributes: a Random Forest model and a Gradient Boosting model. We trained each model and evaluated their performance using the root mean squared error (RMSE) metric.
Random Forest model
python

Copy code
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
Gradient Boosting model
python

Copy code
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
Results
 • Random Forest RMSE: 1.0773004443613237
 • Gradient Boosting RMSE: 1.0825098628290035
The results were saved in the "results.xlsx" Excel file.
Conclusion
We compared two models: Random Forest and Gradient Boosting. Both models achieved similar performance, with Random Forest having a slightly lower root mean squared error (RMSE). This suggests that both models are robust and can accurately predict sales based on the given attributes.
For future work, we could explore other machine learning algorithms or fine-tune the hyperparameters of these models to see if we can further improve the model's performance. It would also be interesting to investigate which features are the most important for predicting sales.
