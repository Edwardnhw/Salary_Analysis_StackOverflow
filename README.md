# Salary Analysis Based on Job Mode & Education Level

**Author**: Hon Wa Ng\
**Date**: October 2024  

## Overview

This project analyzes salary variations across different job modes (**remote vs. in-person**) and education levels (**Bachelor’s, Master’s, and Professional degrees**) using statistical methods such as **Welch’s t-test, ANOVA, and bootstrapping**. The dataset is sourced from **Stack Overflow’s Developer Survey 2024**.

## Objectives

- **Compare salaries** between remote and in-person workers.
- **Evaluate salary trends** across education levels.
- **Perform statistical testing** to determine significance in salary differences.
- **Use bootstrapping** to ensure robust comparisons.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualizing salary trends by **job mode** and **education level**.
- Identifying distributions and potential outliers.

### 2. Hypothesis Testing
- **Welch’s t-test**: Compare remote vs. in-person salaries.
- **ANOVA**: Assess salary differences across education groups.
- **Kolmogorov-Smirnov Test**: Check for normality.
- **Levene’s Test**: Evaluate variance equality.

### 3. Bootstrapping for Robustness
- Generate **10,000 resampled salary means**.
- Compute **pairwise salary differences**.
- Normalize mean differences using **Welch’s t-score**.

### 4. Result Interpretation
- Statistical significance of **job mode and education level** on salary.
- Confidence in findings using **bootstrapped validation**.

## Key Findings

- **Remote vs. In-Person Salaries**:
  - Significant salary differences identified using t-tests and bootstrapping.
  - Bootstrapped results align with the original t-test conclusions.

- **Education Level and Salary**:
  - ANOVA shows statistical significance in salary differences.
  - Higher education levels correlate with increased salary, but variability exists.

## Data Source

- **Stack Overflow Developer Survey 2024** (https://survey.stackoverflow.co/)
- Contains salary, education level, work mode, and demographic information.

## Repository Structure
```bash

Salary_Analysis_StackOverflow/
│── data/
│   │── sample_data.csv  # Reduced dataset
│   │── .DS_Store        # System file (recommended to remove)
│
│── LICENSE              # License for the project
│── reduce_size.ipynb    # Notebook for dataset size reduction
│── salary_analysis.ipynb # Jupyter Notebook for analysis
│── .DS_Store            # Duplicate system file (recommended to remove)

```

### Notes:
- `.DS_Store` files are automatically created by macOS. It is recommended to **remove** them using:
  ```sh
  find . -name ".DS_Store" -delete


---

## Installation & Usage

### 1. Clone the Repository
```
- git clone https://github.com/your-username/Salary_Analysis_StackOverflow.git cd Salary_Analysis_StackOverflow
```

### 2. Install Dependencies
```
pip install -r requirements.txt

```

### 3. Run the Analysis
```
python salary_analysis.py
```


---

## Results Visualization

This project includes multiple visualizations such as:
- Salary trends by experience (Line plot).
- Average salary by education level (Bar chart).
- Job mode salary distributions (Histograms & KDE plots).
- Outlier detection & removal using IQR.
- Bootstrapped confidence intervals & statistical tests.

Sample Output:
- Bootstrapped t-test result: t-statistic: -2.35, p-value: 0.018
- Conclusion: Remote workers earn significantly more than in-person workers.


---

## Data Source

- **Stack Overflow Developer Survey 2024**
- Contains salary, education level, work mode, and demographic information.

---

## Next Steps & Improvements

- Extend analysis to other factors (e.g., programming language, country).
- Implement Machine Learning models for salary prediction.
- Automate data processing with Pandas & NumPy optimizations.

---

## Notes

Remove `.DS_Store` files (macOS system files) before pushing to GitHub  

```
find . -name ".DS_Store" -delete
```

---

