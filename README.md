# OfficerFreddy
Based on a DataCamp project "Credit Card Fraud"
Data Source - https://www.kaggle.com/datasets/kartik2112/fraud-detection

Predicting Credit Card Fraud with OfficerFreddy
By: Anish (Ann) Iyappan - LinkedIn
Background
This is a DataCamp project focused on detecting credit card fraud using Python and SQL.
A new credit card company has launched in the western United States, positioning itself as one of the safest credit cards on the market. To live up to this promise, the company has hired me as a data scientist to identify and stop fraud before it impacts customers.
They provided detailed data on past transactions, each labeled as either fraudulent or legitimate. My mission: build a robust fraud detection system that prioritizes catching fraud, even if it means occasionally flagging some legitimate transactions (false positives).
Motivation
Fraudulent transactions result in significant financial losses and erode customer trust. While some false positives are tolerable, missing actual fraud can have severe consequences. This project balances these trade-offs, with a strong focus on maximizing fraud detection (high recall) while keeping customer inconvenience low.
Data Overview
Data Sets Overview
1.	Train Set (fraudTrain.csv)
Description: Historical transaction data that includes known labels (whether each transaction was fraudulent or not).
Purpose: Used to train the model, learn user spending behaviors, and understand patterns that differentiate fraud from legitimate transactions.
Key columns:
•	cc_num: Credit card number identifier (user ID).
•	amt: Transaction amount.
•	trans_date_trans_time: Timestamp of transaction.
•	merchant: Merchant name.
•	merch_lat, merch_long: Merchant location (latitude and longitude).
•	is_fraud: Target label (1 = fraud, 0 = legit).
•	Other supporting features such as category and location metadata.

2.	Test Set (fraudTest.csv)
Description: Similar structure to the training set but contains new, unseen transactions.
Purpose: Used to evaluate model performance on realistic future data to simulate live deployment conditions.
Key columns: Same as training set, including is_fraud for evaluation.

3.	Feature-Enhanced Sets (train_fe & test_fe)
After feature engineering (via the create_features_safe() function), new columns are added to the original train and test data:
•	amount_deviation
•	hour_deviation
•	speed_kph
•	merchant_risk
•	trans_last_hour
Additional user behavior and geographic features.
Purpose: Enable the model to detect sophisticated fraud patterns (e.g., unusually high amounts, impossible travel speeds).

Feature Engineering
A key highlight of this project is the creation of advanced, domain-inspired features to capture nuanced fraudulent behaviors.
1.	Behavioral Features
•	Average transaction amount and standard deviation per user (cc_num).
•	Amount deviation: difference between transaction amount and user's average.
•	Common transaction hour: typical hour a user transacts, with hour_deviation measuring deviations from this pattern.
 

2.	Geographic Features
Distance traveled between consecutive merchant locations using the Haversine formula.
Speed (kph): derived from travel distance and time between transactions, identifying unrealistic travel behavior.
 

3.	Velocity Features
Transaction count in the last hour: helps detect burst transaction patterns typical of fraud.
These features were engineered via a custom create_features_safe() function to systematically augment the dataset.
 


4.	Temporal Features

Temporal features were also introduced, such as the number of transactions in the last hour for each user, capturing sudden bursts of activity that often indicate fraud.

 


5.	Merchant Risk Features
Merchant risk score: average fraud rate for each merchant, identifying higher-risk merchants.
 
Data Cleaning
•	Replaced infinite values with NaN.
•	Filled missing values with median values from the training data (to prevent leakage).
•	Ensured robust preparation before feeding to the model.
 
 





Selected Features
'amt', 'amount_deviation', 'speed_kph', 'hour_deviation', 'merchant_risk', 'trans_last_hour'
 

Verifying the Featured Engineering Functions
After feature engineering, we conducted careful verification to confirm that the new features were correctly created, and that no data was lost or corrupted. We printed summary shapes, previews of engineered columns such as amount_deviation and speed_kph, and descriptive statistics to validate ranges and distributions.
 

Modeling Approach
A Random Forest Classifier was used because of its strong performance on tabular data and interpretability.
•	n_estimators: 100
•	class_weight: {0: 1, 1: 10} to emphasize catching fraud.
•	random_state: 42 for reproducibility.
 

Model Performance & Evaluation
Feature Importance
After training, we conducted a feature importance analysis to understand which factors most strongly influence fraud predictions.
 
The analysis revealed that the transaction amount (amt) and the deviation from a user's typical amount (amount_deviation) were the most significant drivers. 
•	Top contributing features: amt (transaction amount). amount_deviation, hour_deviation
•	Less influential: speed_kph, trans_last_hour, merchant_risk.
 

Threshold Optimization & Trade-off Analysis
After evaluating model probabilities, we manually chose an optimized threshold of 0.30. At this point, the model achieved a recall of approximately 61%, successfully identifying most fraud cases, while precision was around 19%, meaning that about one in five flagged transactions was truly fraudulent.
This decision to lower the threshold from the default 0.5 is strategic: it prioritizes catching more fraud (high recall) even at the cost of more false alarms, which aligns with the business priority of minimizing missed fraud.
The area under the precision-recall curve (PR AUC) was approximately 0.41, supporting that the model maintains a reasonable balance across thresholds.
 
 

Confusion Matrix & Detailed Threshold Analysis
Approach
In this section, we further assess the model’s classification results using a confusion matrix, evaluated at an optimized threshold (0.30). The confusion matrix breaks down predictions into four key categories:

•	True Positives (TP): Fraudulent transactions correctly identified as fraud.
•	True Negatives (TN): Legitimate transactions correctly identified as legitimate.
•	False Positives (FP): Legitimate transactions incorrectly flagged as fraud, potentially impacting customer experience.
•	False Negatives (FN): Fraudulent transactions that were missed, representing residual fraud risk.

Metrics Computed
•	Recall (Sensitivity): Proportion of actual fraud cases correctly detected.
•	Precision: Proportion of flagged transactions that are actually fraudulent.
•	False Positive Rate: Proportion of legitimate transactions mistakenly flagged as fraud.
 
Performance Metrics
To evaluate the model's ability to distinguish fraudulent from non-fraudulent transactions, two key performance curves were analyzed: the Receiver Operating Characteristic (ROC) curve and the Precision-Recall (PR) curve.
ROC Curve Analysis
The ROC curve plots the true positive rate (fraud correctly identified) against the false positive rate (legitimate transactions incorrectly flagged as fraud) across various threshold settings.
Precision-Recall Curve Analysis
The PR curve focuses on the trade-off between precision (the proportion of flagged transactions that are truly fraud) and recall (the proportion of actual fraud caught).
 
 

 
The ROC AUC (Area Under Curve) score achieved by this model is 94.3%, indicating strong discriminatory power.
A high ROC AUC suggests that the model is generally effective at distinguishing between fraudulent and genuine transactions, regardless of the chosen threshold.
The PR AUC score is 40.0%, which is lower than the ROC AUC.
This lower value indicates that while the model can separate fraud well overall, its ability to maintain high precision when striving for higher recall is limited.
In fraud detection, recall is often prioritized over precision, as missing fraud carries higher business risk than occasionally flagging legitimate transactions.
Threshold & Cost Analysis
Business Metrics & Threshold Impact Analysis
Approach
In this section, we evaluated the model’s real-world performance using business-centric metrics. By applying a chosen probability threshold (here, 0.30), we transformed predicted probabilities into a final decision on whether each transaction is flagged as fraud.
The business_metrics function was defined to compute:
•	Fraud caught percentage (fraud_caught_pct): Proportion of actual fraud transactions successfully identified.
•	False positive percentage (false_positives_pct): Proportion of legitimate transactions incorrectly flagged as fraud.
•	Total dollars protected (dollars_protected): Sum of transaction amounts for predicted fraud cases, representing the potential monetary value safeguarded.
•	Number of fraud cases missed (fraud_missed): Count of fraud cases that were not captured by the model.
 





Behavioral Insights & Fraud Patterns
Understanding the behavior of key features is critical to interpreting model decisions and designing effective fraud prevention strategies. We conducted an in-depth analysis of four primary features:

•	Transaction Amount (amt)
•	Amount Deviation (amount_deviation)
•	Speed in km/h (speed_kph)
•	Hour Deviation (hour_deviation)

Distribution Patterns
We visualized these features using distribution plots (density plots), separated by fraud and non-fraud transactions. The distributions provide valuable insights into how fraud behavior deviates from typical patterns:

1.	Transaction Amount (amt): Fraudulent transactions generally exhibit significantly higher transaction amounts, indicating that fraudsters often attempt to maximize gains per fraudulent activity.

2.	Amount Deviation: This feature captures the deviation of a transaction from a user's typical spending behavior. Fraudulent transactions showed large positive deviations, reflecting unusual or unexpected amounts relative to individual historical spending.

3.	Hour Deviation: Fraudulent transactions tend to occur at times that deviate from a user's most frequent transaction hours. This indicates that time-based anomalies are strong fraud signals.

4.	Speed (speed_kph): While theoretically a strong feature (measuring velocity between transactions to detect impossible travel), in practice it did not show significant separation in our dataset, potentially due to data sparsity or missing prior merchant locations.
About the system:
•	The system prioritizes fraud detection to safeguard customer assets.
•	Business cost simulation confirms that the selected threshold maximizes net savings.
•	Model design and feature engineering enable interpretability and actionable risk segmentation.

Median-Based Fraud Ratios
To further quantify these behavioral patterns, we calculated the median values of each feature for both fraud and non-fraud transactions, and derived fraud-to-legit ratios:
Feature	Median (Non-Fraud)	Median (Fraud)	Fraud-to-Legit Ratio
amt	~47

	~397

	8.4x


hour_deviation	~9

	~1

	0.1x


amount_deviation	~-23

	~326

	-14.1x


speed_kph	0	0	N/A

 
 

Cost Analysis
Cost-Benefit Analysis and Threshold Optimization
A critical component of deploying a fraud detection model in practice is to balance the trade-off between catching fraudulent transactions and minimizing operational costs, including false positive reviews and missed fraud. To achieve this, we analyzed different decision thresholds using a cost-based framework.
Framework
We define the following costs and assumptions:
•	Fraud cost multiplier: Fraudulent transactions have an estimated cost impact 1.5× their value (reflecting chargebacks, penalties, and investigation costs).
•	False positive review cost: Each flagged legitimate transaction incurs an operational review cost of $10.
•	Using these definitions, we evaluated different probability thresholds to estimate:
•	Fraud caught value: The estimated value of fraud transactions successfully detected and prevented.
•	Fraud missed cost: The estimated cost of fraud transactions that go undetected.
•	False positive cost: The operational cost of reviewing flagged legitimate transactions.
•	Net savings: Total savings calculated as fraud caught value minus fraud missed cost and false positive cost.
 
 
 

Challenges Addressed
During this project, several practical data science and engineering challenges were encountered:
1.	Handling Missing and Noisy Data:
Some columns had missing values or unexpected placeholders (like dashes or 'missing' strings). We addressed this by systematically replacing them with median values or default placeholders to maintain model stability.
2.	Engineering Behavioral and Geographic Features:
Designing and integrating new features such as transaction speed (km/h), deviation from a user’s average spend, and rolling transaction counts required careful data transformation and grouping logic.
3.	Dealing with Infinite Values:
During feature engineering, calculations like speed (distance over time) could result in infinite or undefined values (for example, if the time difference between transactions was zero).
These infinite values needed to be replaced thoughtfully, often using the median of that feature or setting them to zero, to prevent model training failures and ensure accurate downstream predictions.
4.	Balancing Precision and Recall:
Fraud detection typically prioritizes recall (catching more fraud) over precision, which required tuning thresholds and interpreting trade-offs carefully.
5.	Model Interpretability vs Performance:
While advanced features improved model accuracy, they also made the model slightly less interpretable for non-technical stakeholders. Balancing complexity and explainability was a key consideration.

Conclusion
This project addressed the executive’s goal of developing a robust fraud detection system for a new credit card company entering the western U.S. market, with a strong emphasis on customer safety and cautious risk management.
Our final model leverages advanced behavioral, temporal, and geographic features — such as deviation from typical spending patterns and travel speed between transactions — to strengthen fraud detection capabilities. The model was carefully tuned to prioritize recall, aligning with the executive’s directive that it is better to flag more transactions as potentially fraudulent than to miss actual fraud.

Key outcomes include:
•	High fraud capture rate: The model successfully identifies a significant proportion of fraudulent transactions, ensuring strong protection for customers.
•	Flexible threshold tuning: The model's decision threshold can be adjusted to favor higher recall when necessary, providing business flexibility as risk tolerance evolves.
•	Cost and impact balance: Detailed cost-benefit analyses confirmed that the selected threshold delivers optimal net savings while minimizing operational review costs.
•	Transparent risk segmentation: Clear risk categorization supports operational teams in prioritizing investigations and enhances trust in model-driven decisions.

Overall, the model demonstrates strong potential for production deployment and serves as a solid foundation for ongoing enhancements. It aligns well with the company’s brand promise of safety and positions it competitively as a secure choice for consumers.
<img width="468" height="633" alt="image" src="https://github.com/user-attachments/assets/677e471c-7ef1-4fdf-91fb-d1b3ea1b2a34" />
