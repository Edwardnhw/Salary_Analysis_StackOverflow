# Salary Analysis Based on Job Mode & Education Level

**Author**: Hon Wa Ng  
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

- **Stack Overflow Developer Survey 2024**
- Contains salary, education level, work mode, and demographic information.

## Repository Structure

Salary_Analysis_StackOverflow/
│── data/
│   │── sample_data.csv  # Reduced dataset
│   │── .DS_Store        # System file (recommended to remove)
│
│── LICENSE              # License for the project
│── reduce_size.ipynb    # Notebook for dataset size reduction
│── salary_analysis.ipynb # Jupyter Notebook for analysis
│── .DS_Store            # Duplicate system file (recommended to remove)

### Notes:
- `.DS_Store` files are automatically created by macOS. It is recommended to **remove** them using:
  ```sh
  find . -name ".DS_Store" -delete



