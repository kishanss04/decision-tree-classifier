# Decision Tree Classifier 

## Project Overview
This project involves building a **Decision Tree Classifier** to predict whether a customer will purchase a product or service. The prediction is based on demographic and behavioral data using the **Bank Marketing Dataset** from the UCI Machine Learning Repository.

---

## Dataset
- **Dataset Name**: Bank Marketing Dataset
- **Source**: UCI Machine Learning Repository
- **Description**: The dataset contains marketing campaign data for a Portuguese bank. It includes information about customer demographics, contact details, and the outcome of a direct marketing campaign.
- **Download Link**: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## Objective
To classify whether a customer will respond positively (`yes`) or negatively (`no`) to a marketing offer based on their demographic and behavioral data.

---

## Project Workflow
1. **Data Loading and Exploration**:
   - Load the dataset and check for missing values or inconsistencies.
   - Perform exploratory data analysis (EDA) to understand trends and patterns.

2. **Data Preprocessing**:
   - Handle missing values and encode categorical variables.
   - Split the dataset into training and testing sets.

3. **Model Development**:
   - Build a Decision Tree Classifier using `scikit-learn`.
   - Tune hyperparameters to optimize model performance.

4. **Model Evaluation**:
   - Evaluate the model using metrics such as accuracy, precision, recall, and F1 score.
   - Generate a confusion matrix and visualize the decision tree.

---

## Project Structure
```bash
   decision-tree-classifier/
├── data/
│   ├── bank-marketing-dataset.csv      # Original dataset file
├── src/
│   ├── decision_tree_classifier.py     # Main script for building and evaluating the model
├── outputs/
│   ├── decision_tree_model             # Serialized decision tree model
├── README.md                           # Project documentation
```
---

## Results
- **Model Accuracy**: `XX%` (replace with your model's accuracy)
- **Insights**:
  - Customers with certain professions and contact types had higher positive responses.
  - Age, marital status, and campaign outcome were significant factors.

---

## Requirements
Install the necessary Python libraries using:
```bash
pip install -r requirements.txt
```
## How to Run
1. Clone this repository
   ```bash
   git clone https://github.com/kishanss04/decision-tree-classifier.git
   cd decision-tree-classifier
   ```
2. Run the decision_tree_classifier.py script:
   ```bash
   python src/decision_tree_classifier.py
   ```


    ## Troubleshooting
If you encounter any issues while running the project, feel free to leave a comment on the GitHub repository, or you can contact me directly at:

Email: kishanss1804@gmail.com
GitHub: kishanss04

## License
This project is licensed under the MIT License.

