# %% [markdown]
# # Salary Analysis Based on Job Mode & Education Level
# **Author**: Hon Wa Ng
# **Date**: Oct 2024  
# 
# ≈ Project Overview
# This project analyzes salary differences between **remote vs. in-person workers** and across **education levels** using statistical tests such as Welch’s t-test, ANOVA, and bootstrapping. The dataset is sourced from **Stack Overflow’s Developer Survey 2024**.
# 
# ## Structure
# 1. **Exploratory Data Analysis (EDA)**
# 2. **Hypothesis Testing (Welch’s t-test, ANOVA)**
# 3. **Bootstrapping for Robustness**
# 4. **Results & Interpretation**
# 

# %%
# Import necessary libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Use the detected encoding
df = pd.read_csv('data/sample_data.csv')

# Display the first few rows of the dataset 
df.head()

# Check the column names and data types
df.info()


# %% [markdown]
# ## Exploratory Data Analysis: Visualizing Trends in Salary and Experience
# 
# This section explores how **salary trends evolve with increasing years of professional coding experience**.  
# A **line plot** is used to analyze whether salaries **grow, plateau, or decline** as experience increases.
# 
# ### Steps:
# 1. **Group the data** by `YearsCodePro` to compute the **average salary** for each experience level.
# 2. **Visualize the trend** using a **line plot**, with:
#    - `YearsCodePro` on the x-axis (representing years of experience).
#    - `ConvertedCompYearly` on the y-axis (representing average salary).
#    - **Markers** for better visibility of data points.
#    - **Grid lines** to improve readability.
# 
# ### Key Insights:
# - The visualization helps identify **whether higher experience consistently leads to higher salaries**.
# - Certain experience levels may have **unexpected salary fluctuations** due to industry demand, career shifts, or other factors.
# - The trend can indicate **salary ceilings** where income growth slows down after a certain number of years.
# 
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group data by 'YearsCodePro' and calculate the average salary for each experience level
salary_trends = df.groupby('YearsCodePro')['ConvertedCompYearly'].mean().reset_index()

# Plot the line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='YearsCodePro', y='ConvertedCompYearly', data=salary_trends, marker='o', color='b')
plt.title('Salary Trends Over Years of Professional Coding Experience')
plt.xlabel('Years of Professional Coding Experience')
plt.ylabel('Average Salary (Annual Income)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()


# %% [markdown]
# ## Visualizing Salary Trends Across Different Education Levels
# 
# This section explores how **education level impacts average annual salary**.  
# A **bar plot** is used to present the relationship between different education levels and salaries, helping to **identify patterns in earnings** based on educational background.
# 
# ### Steps:
# 1. **Load the dataset** and filter relevant columns.
# 2. **Group salaries by education level** and compute the average salary.
# 3. **Sort education levels** in a logical order for better readability.
# 4. **Plot a bar chart** to visualize salary trends.
# 
# ### Key Insights:
# - The analysis helps in understanding **whether higher education levels correlate with higher salaries**.
# - Certain educational paths might **have outliers or unexpected trends** that require further investigation.
# 

# %%

# Group by education level and calculate the average salary for each level
avg_salary_by_ed_level = df.groupby('EdLevel')['ConvertedCompYearly'].mean().reset_index()

# Sort the education levels to maintain a logical order
education_order = ['Primary/elementary school', 'Secondary school', 
                   'Some college/university study without earning a degree', 
                   'Associate degree', 'Bachelor’s degree', 'Professional degree', 
                   'Master’s degree', 'Something else']
avg_salary_by_ed_level['EdLevel'] = pd.Categorical(avg_salary_by_ed_level['EdLevel'], 
                                                   categories=education_order, ordered=True)
avg_salary_by_ed_level = avg_salary_by_ed_level.sort_values('EdLevel')

# Create the bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='EdLevel', y='ConvertedCompYearly', data=avg_salary_by_ed_level)

# Set plot title and labels
plt.title('Average Salary by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Average Salary (Annual Income)')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Distribution of Employment Types: Remote, In-Person, Hybrid
# 
# This section analyzes the **distribution of different work modes** among respondents, categorizing them as **remote, in-person, or hybrid workers**. A **pie chart** is used to visualize the proportion of each work mode.
# 
# ### Steps:
# 1. **Load the dataset** and extract the `RemoteWork` column.
# 2. **Count the occurrences** of each work mode category.
# 3. **Format labels** to display both **percentage and count** of respondents.
# 4. **Plot a pie chart** to illustrate the distribution.
# 
# ### Key Insights:
# - The visualization helps to understand **how prevalent remote work is compared to in-person or hybrid work**.
# - The proportion of work modes can indicate **shifts in industry trends**, such as a rise in remote work.
# - The chart provides insights into **work flexibility and job preferences** in the tech industry.
# 

# %%


# Count the occurrences of each job mode (remote, in-person, hybrid)
work_mode_counts = df['RemoteWork'].value_counts()

labels = work_mode_counts.index
sizes = work_mode_counts.values

# to display both percentage and count
def autopct_format(pct, allvals):
    absolute = int(pct/100. * sum(allvals))
    return f"{pct:.1f}%\n({absolute:d})"

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct=lambda pct: autopct_format(pct, sizes), startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Distribution of Employment Types (Remote, In-Person, Hybrid)')
plt.axis('equal') 
plt.show()


# %% [markdown]
# ## Detecting and Removing Outliers Using the IQR Method
# 
# This step involves **identifying and removing outliers** in salary data using the **Interquartile Range (IQR) method**.  
# The dataset is divided into **remote** and **in-person** job modes to compare salary distributions.
# 
# ### Steps:
# 1. **Load the dataset** and filter salaries by job mode (`Remote` vs. `In-Person`).
# 2. **Apply the IQR method** to detect and remove outliers:
#    - Calculate **Q1 (25th percentile)** and **Q3 (75th percentile)**.
#    - Compute **IQR (Q3 - Q1)**.
#    - Determine **lower and upper bounds**:  
#      - `Lower Bound = Q1 - 1.5 * IQR`
#      - `Upper Bound = Q3 + 1.5 * IQR`
#    - Filter salaries within this range.
# 3. **Compute descriptive statistics** (mean, median, standard deviation).
# 4. **Compare cleaned salary distributions** for both groups.
# 
# ### Key Insights:
# - Removing outliers ensures that **extreme values do not skew the salary analysis**.
# - The comparison of **mean, median, and standard deviation** helps identify salary trends across different work modes.
# - The number of valid salary entries **before and after outlier removal** provides insight into data variability.
# 

# %%
in_person_salaries = df[df['RemoteWork'] == 'In-person']['ConvertedCompYearly']
remote_salaries = df[df['RemoteWork'] == 'Remote']['ConvertedCompYearly']

# to remove outliers using IQR and identify them
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers from both groups and report
in_person_salaries_cleaned = remove_outliers_and_report(in_person_salaries)

remote_salaries_cleaned = remove_outliers_and_report(remote_salaries)

# Compute descriptive statistics
in_person_mean = in_person_salaries_cleaned.mean()
in_person_median = in_person_salaries_cleaned.median()
in_person_std = in_person_salaries_cleaned.std()

remote_mean = remote_salaries_cleaned.mean()
remote_median = remote_salaries_cleaned.median()
remote_std = remote_salaries_cleaned.std()

# Print the descriptive statistics for In-Person jobs
print("\nDescriptive Statistics for In-Person Job Mode:")
print(f"Mean Salary: {in_person_mean:.2f}")
print(f"Median Salary: {in_person_median:.2f}")
print(f"Standard Deviation: {in_person_std:.2f}")
print(f"Number of Entries: {len(in_person_salaries_cleaned)}")

# Print the descriptive statistics for Remote jobs
print("\nDescriptive Statistics for Remote Job Mode:")
print(f"Mean Salary: {remote_mean:.2f}")
print(f"Median Salary: {remote_median:.2f}")
print(f"Standard Deviation: {remote_std:.2f}")
print(f"Number of Entries: {len(remote_salaries_cleaned)}")


# %% [markdown]
# ## Performing a Two-Sample T-Test After Removing Outliers
# 
# This step conducts a **two-sample t-test** to compare the **mean salaries** of remote vs. in-person workers after removing outliers.
# 
# ### Steps:
# 1. **Load and clean the data**  
#    - Filter salaries by `RemoteWork` status.
#    - Remove outliers using the **Interquartile Range (IQR) method**.
# 
# 2. **Prepare for the t-test**  
#    - Compute **sample sizes** for both groups.
#    - Calculate **means** and **variances** for remote and in-person salaries.
# 
# 3. **Perform a two-sample t-test**  
#    - The t-test helps determine whether the salary difference between **remote** and **in-person** workers is statistically significant.
# 
# ### Key Insights:
# - **Hypothesis Testing**  
#   - **Null Hypothesis (H₀)**: There is no significant difference in salaries between remote and in-person workers.  
#   - **Alternative Hypothesis (H₁)**: There is a significant difference in salaries between the two groups.
# 
# - **Why Remove Outliers?**  
#   - Outliers can distort statistical results, making it harder to **detect real differences**.
#   - Cleaning the data ensures **valid and reliable** hypothesis testing.
# 
# - **Next Steps:**  
#   - Run the full t-test with `scipy.stats.ttest_ind()` to formally test the hypothesis.
#   - Analyze the **p-value** to determine statistical significance.
# 

# %%
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro

in_person_salaries = df[df['RemoteWork'] == 'In-person']['ConvertedCompYearly']
remote_salaries = df[df['RemoteWork'] == 'Remote']['ConvertedCompYearly']

# SRemove outliers using the IQR method (already implemented)
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Number of outliers removed: {len(outliers)}")
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers 
in_person_cleaned = remove_outliers_and_report(in_person_salaries)
remote_cleaned = remove_outliers_and_report(remote_salaries)

# Manual two-sample t-test calculation
# Sample sizes
n1 = len(remote_cleaned)
n2 = len(in_person_cleaned)

# Means
mean1 = np.mean(remote_cleaned)
mean2 = np.mean(in_person_cleaned)

# Variances
var1 = np.var(remote_cleaned, ddof=1)  # ddof=1 ensures sample variance
var2 = np.var(in_person_cleaned, ddof=1)

# Pooled standard deviation
pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
print(f"Pooled Standard Deviation: {pooled_std}")

# t-statistic calculation (using weltch as we cannot assume the variances of 2 groups to be equal )
t_manual_welch = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))

# Degrees of freedom
df_welch = n1 + n2 - 2  # Avoid overwriting the DataFrame 'df'
print(f"Manual t-value: {t_manual_welch}, Degrees of Freedom: {df_welch}")


# Built-in t-test using scipy
t_stat, p_value = stats.ttest_ind(remote_cleaned, in_person_cleaned, equal_var=False)

# Print Python's built-in t-test result
print(f"Built-in t-test t-value: {t_stat}, p-value: {p_value}")

# Checking assumptions
# Normality assumption
print("\nChecking for normality using Shapiro-Wilk test:")
stat_in_person, p_in_person = shapiro(in_person_cleaned)
stat_remote, p_remote = shapiro(remote_cleaned)

print(f"In-Person group: W={stat_in_person}, p-value={p_in_person}")
print(f"Remote group: W={stat_remote}, p-value={p_remote}")

if p_in_person > 0.05 and p_remote > 0.05:
    print("Both groups follow a normal distribution (p > 0.05).")
else:
    print("At least one group does not follow a normal distribution (p <= 0.05).")

# Equal variance assumption
stat_levene, p_levene = stats.levene(in_person_cleaned, remote_cleaned)

print(f"\nLevene's test for equal variances: W={stat_levene}, p-value={p_levene}")

if p_levene > 0.05:
    print("Equal variance assumption holds (p > 0.05).")
else:
    print("Equal variance assumption does not hold (p <= 0.05).")

# Independence assumption
print("Checking normality using Shapiro-Wilk test:")
shapiro_in_person = stats.shapiro(in_person_cleaned)
shapiro_remote = stats.shapiro(remote_cleaned)

print(f"In-person: W={shapiro_in_person.statistic}, p-value={shapiro_in_person.pvalue}")
print(f"Remote: W={shapiro_remote.statistic}, p-value={shapiro_remote.pvalue}")

if shapiro_in_person.pvalue > 0.05 and shapiro_remote.pvalue > 0.05:
    print("Both groups are approximately normally distributed.")
else:
    print("At least one group is not normally distributed.")


# %% [markdown]
# ## Bootstrapping and T-Test for Salary Comparison
# 
# This step performs **bootstrapping and a two-sample t-test** to compare salary distributions between **remote** and **in-person** workers.
# 
# ### Steps:
# 1. **Clean the Data:**
#    - Remove outliers using the **Interquartile Range (IQR) method**.
#    - Filter salaries based on work mode (**Remote vs. In-Person**).
# 
# 2. **Bootstrapping for Statistical Robustness:**
#    - Perform **10,000 resampling iterations** with replacement.
#    - Compute **mean salary estimates** for both job modes.
# 
# 3. **Normalize Salary Differences:**
#    - Calculate **variance and sample sizes** for both groups.
#    - Compute **Welch's t-score** to assess salary variations.
# 
# 4. **Conduct T-Tests:**
#    - **Bootstrapped t-test**: Uses resampled salary distributions.
#    - **Regular t-test**: Compares original cleaned salary distributions.
# 
# ### Key Insights:
# - **Why Bootstrapping?**
#   - Bootstrapping provides **more robust mean estimates** by resampling from observed data.
#   - Helps in handling **small sample sizes** and improving statistical reliability.
# 
# - **Hypothesis Testing:**
#   - **Null Hypothesis (H₀)**: No significant salary difference between **remote** and **in-person** workers.
#   - **Alternative Hypothesis (H₁)**: There is a significant difference in salaries.
# 
# - **Next Steps:**
#   - Analyze the **p-values** from both t-tests.
#   - Interpret the results to determine **whether remote workers earn significantly more or less** than in-person employees.
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


in_person_salaries = df[df['RemoteWork'] == 'In-person']['ConvertedCompYearly']
remote_salaries = df[df['RemoteWork'] == 'Remote']['ConvertedCompYearly']

# SRemove outliers using the IQR method
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Number of outliers removed: {len(outliers)}")
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers 
in_person_cleaned = remove_outliers_and_report(in_person_salaries)
remote_cleaned = remove_outliers_and_report(remote_salaries)

# Bootstrap parameters
n_replications = 10000

# Bootstrap sampling and mean calculation for in-person
in_person_bootstrap_means = [
    np.mean(np.random.choice(in_person_cleaned, size=len(in_person_cleaned), replace=True)) for _ in range(n_replications)
]

# Bootstrap sampling and mean calculation for remote
remote_bootstrap_means = [
    np.mean(np.random.choice(remote_cleaned, size=len(remote_cleaned), replace=True)) for _ in range(n_replications)
]

# Calculate the difference in means
diff_in_means =  np.array(remote_bootstrap_means) - np.array(in_person_bootstrap_means)
Norm_Bootstrap_mean_diff = diff_in_means / np.sqrt(np.var(in_person_bootstrap_means)/len(in_person_bootstrap_means) + np.var(remote_bootstrap_means)/len(remote_bootstrap_means))


# Normalize the difference in means based on Welch's t-score
var_in_person = np.var(in_person_cleaned, ddof=1)
var_remote = np.var(remote_cleaned, ddof=1)
n_in_person = len(in_person_cleaned)
n_remote = len(remote_cleaned)

se_diff = np.sqrt(var_in_person/n_in_person + var_remote/n_remote)
t_score_normalization = diff_in_means / se_diff

# Print Welch's Normalization Mean T-Score
mean_t_score_normalization = np.mean(t_score_normalization)
print("Welch's Normalization Mean T-Score:", mean_t_score_normalization)

# Perform t-test using bootstrapped means
t_stat_bootstrap, p_value_bootstrap = stats.ttest_ind(remote_bootstrap_means, in_person_bootstrap_means, equal_var=False)

# Perform regular t-test for comparison (before bootstrapping)
t_stat_original, p_value_original = stats.ttest_ind(remote_cleaned, in_person_cleaned, equal_var=False)

print(f"Bootstrap t-test: t = {t_stat_bootstrap}, p = {p_value_bootstrap}")
print(f"Original t-test: t = {t_stat_original}, p = {p_value_original}")

# %% [markdown]
# ## Visualizing Bootstrapped and Normalized Salary Distributions
# 
# This section presents three key visualizations comparing remote vs. in-person salaries.
# 
# ### Figure 1: Bootstrapped Salary Distributions
# - **Compares bootstrapped means** for remote and in-person workers.
# - **Histogram with KDE curves** shows salary distributions.
# 
# ### Figure 2: Difference in Means
# - **Visualizes the distribution** of salary differences (Remote - In-Person).
# - Helps assess **variability and confidence** in mean salary differences.
# 
# ### Figure 3: Normalized Difference in Means
# - **Standardizes salary differences** using Welch’s t-score.
# - Evaluates **statistical significance** of salary gaps.
# 
# These plots provide insights into **wage disparities and data reliability** across work modes.
# 

# %%

# Plotting the bootstrapped distributions and the difference in means

# Figure 1: Bootstrapped distributions for in-person and remote job modes
plt.figure(figsize=(10, 6))
sns.histplot(in_person_bootstrap_means, color='blue', label='In-Person', kde=True, stat='count')
sns.histplot(remote_bootstrap_means, color='orange', label='Remote', kde=True, stat='count')
plt.title("Bootstrapped Distributions for In-Person and Remote Job Modes")
plt.xlabel("Bootstrapped Mean Salary (Annual Income)")
plt.ylabel("Count")
plt.legend()
plt.show()

# %%
# Figure 2: Distribution of the difference in means
plt.figure(figsize=(10, 6))
sns.histplot(Norm_Bootstrap_mean_diff, kde=True, color='purple', stat='count')
plt.title("Distribution of the Difference in Means")
plt.xlabel("Difference in Mean Salary (Remote - In-Person)")
plt.ylabel("Count")
plt.show()

# %%
# Figure 3: Normalized distribution of the difference in means
observed_mean_diff =  np.mean(remote_cleaned) - np.mean(in_person_cleaned)
normalized_diffs = (diff_in_means - observed_mean_diff) / np.std(diff_in_means)

plt.figure(figsize=(10, 6))
sns.histplot(t_score_normalization, kde=True, color='red', stat='count')
plt.title("Normalized Distribution of the Difference in Means")
plt.xlabel("Normalized Difference in Mean Salary")
plt.ylabel("Count")
plt.show()

# %% [markdown]
# ## Bootstrapped T-Test for Salary Comparison
# 
# This section applies **bootstrapping** and a **t-test** to compare remote vs. in-person salaries.
# 
# ### Key Steps:
# 1. **Clean Data** – Remove outliers using the IQR method.
# 2. **Bootstrap Sampling** – Generate 10,000 resampled salary means.
# 3. **T-Test Analysis**  
#    - Perform a **bootstrapped t-test** on resampled means.
#    - Compare with the **manual t-test** from Q2b.
# 4. **Hypothesis Evaluation**  
#    - Reject **H₀** if `p-value < 0.05`, indicating salary differences.
# 
# ### Results:
# - If `p-value_bootstrap` is **low**, there is **stronger evidence** of salary differences.
# - **Comparison with Q2b** helps validate the results' robustness.
# 
# This ensures **statistical reliability** in comparing salary distributions.
# 

# %%
import numpy as np
import scipy.stats as stats

in_person_salaries = df[df['RemoteWork'] == 'In-person']['ConvertedCompYearly']
remote_salaries = df[df['RemoteWork'] == 'Remote']['ConvertedCompYearly']

# Remove outliers using the IQR method (already implemented)
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Number of outliers removed: {len(outliers)}")
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers 
in_person_cleaned = remove_outliers_and_report(in_person_salaries)
remote_cleaned = remove_outliers_and_report(remote_salaries)

# Bootstrap parameters
n_replications = 10000

# Bootstrap sampling and mean calculation for in-person
in_person_bootstrap_means = [
    np.mean(np.random.choice(in_person_cleaned, size=len(in_person_cleaned), replace=True)) for _ in range(n_replications)
]

# Bootstrap sampling and mean calculation for remote
remote_bootstrap_means = [
    np.mean(np.random.choice(remote_cleaned, size=len(remote_cleaned), replace=True)) for _ in range(n_replications)
]


# Perform a two-sample t-test on the bootstrapped means
t_stat_bootstrap, p_value_bootstrap = stats.ttest_ind(remote_bootstrap_means, in_person_bootstrap_means, equal_var=False)

# Print the t-statistic and p-value for the bootstrapped test
print(f"Bootstrapped t-test result:")
print(f"t-statistic: {t_stat_bootstrap}, p-value: {p_value_bootstrap}")

# Set the significance level
alpha = 0.05

# Check if p-value is less than the significance level (0.05)
if p_value_bootstrap < alpha:
    print("Reject the null hypothesis: There is a significant difference in mean salaries between in-person and remote job modes (bootstrapped).")
else:
    print("Fail to reject the null hypothesis: No significant difference in mean salaries between in-person and remote job modes (bootstrapped).")

# Compare the results from Q2b (manual calculation) with Q2d (bootstrapped)
# t_stat_original and p_value_original from Q2b
print(f"Comparison with Q2b (Manual t-test result):")
print(f"Manual t-test: t-statistic = {t_stat_original}, p-value = {p_value_original}")

if p_value_bootstrap < p_value_original:
    print("The bootstrapped p-value is smaller, indicating a stronger rejection of the null hypothesis.")
else:
    print("The manual p-value is smaller or equal, indicating no stronger evidence of rejecting the null hypothesis in the bootstrapped version.")


# %% [markdown]
# ## Outlier Removal and Descriptive Statistics by Education Level
# 
# This section analyzes salaries across **Bachelor’s, Master’s, and Professional degrees** while removing outliers.
# 
# ### Key Steps:
# 1. **Filter Data** – Select only the three education levels.
# 2. **Remove Outliers** – Apply the **IQR method** to each education group.
# 3. **Compute Descriptive Statistics**:
#    - **Mean & Median Salary**
#    - **Standard Deviation**
#    - **Number of Entries (Valid Data Points)**
# 
# ### Results:
# - Provides a **cleaner salary distribution** by education level.
# - Helps assess whether **higher education correlates with salary increases**.
# 
# This step ensures **data consistency** before further analysis.
# 

# %%
import pandas as pd


# Filter the data for the three education levels
education_levels = ['Bachelor’s degree', 'Master’s degree', 'Professional degree']
df_filtered = df[df['EdLevel'].isin(education_levels)]

# Function to remove outliers using IQR method and report them
def remove_outliers(data):
    Q1 = data['ConvertedCompYearly'].quantile(0.25)
    Q3 = data['ConvertedCompYearly'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data['ConvertedCompYearly'] >= lower_bound) & (data['ConvertedCompYearly'] <= upper_bound)]

# Apply outlier removal for each education level using a defined function
def apply_outlier_removal(group):
    return remove_outliers(group)

df_filtered_cleaned = df_filtered.groupby('EdLevel', group_keys=False).apply(
    lambda group: apply_outlier_removal(group[['ConvertedCompYearly']]),include_groups=False
)

# Restore the 'EdLevel' column after the group operation
df_filtered_cleaned = df_filtered_cleaned.join(df_filtered[['EdLevel']], how='left')

# Calculate descriptive statistics (mean, median, std) for each group
descriptive_stats = df_filtered_cleaned.groupby('EdLevel').agg(
    Mean_Salary=('ConvertedCompYearly', 'mean'),
    Median_Salary=('ConvertedCompYearly', 'median'),
    Standard_Deviation=('ConvertedCompYearly', 'std'),
    Number_of_Entries=('ConvertedCompYearly', 'count')
)

# Print the descriptive statistics
print(descriptive_stats)



# %% [markdown]
# ## ANOVA Test for Salary Differences by Education Level
# 
# This section applies **ANOVA** to test for salary differences among **Bachelor’s, Master’s, and Professional degrees**.
# 
# ### Key Steps:
# 1. **Filter Data** – Select salaries for the three education levels.
# 2. **Remove Outliers** – Apply the **IQR method** to clean data.
# 3. **Perform ANOVA**  
#    - Tests if at least one education group has a significantly different mean salary.
#    - If **p-value < 0.05**, salary differences exist.
# 
# 4. **Check ANOVA Assumptions**  
#    - **Normality (Kolmogorov-Smirnov Test)**: Ensures data follows a normal distribution.
#    - **Equal Variance (Levene’s Test)**: Confirms similar salary variance across groups.
# 
# ### Results:
# - If assumptions hold, ANOVA results are valid.
# - If violated, consider **non-parametric alternatives** (e.g., Kruskal-Wallis test).
# 
# This step evaluates whether **education level impacts salary differences** statistically.
# 

# %%
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest

# Filter the data for the three education levels
bachelors_salaries = df[df['EdLevel'] == 'Bachelor’s degree']['ConvertedCompYearly']
masters_salaries = df[df['EdLevel'] == 'Master’s degree']['ConvertedCompYearly']
professional_salaries = df[df['EdLevel'] == 'Professional degree']['ConvertedCompYearly']

education_levels = ['Bachelor’s degree', 'Master’s degree', 'Professional degree']
df_filtered = df[df['EdLevel'].isin(education_levels)]


# SRemove outliers using the IQR method 
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Number of outliers removed: {len(outliers)}")
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers 
bachelors_salaries_cleaned = remove_outliers_and_report(bachelors_salaries)
masters_salaries_cleaned = remove_outliers_and_report(masters_salaries)
professional_salaries_cleaned = remove_outliers_and_report(professional_salaries)


# Perform ANOVA
anova_result = stats.f_oneway(bachelors_salaries_cleaned, masters_salaries_cleaned, professional_salaries_cleaned)

# Print ANOVA results
print("ANOVA F-statistic: {}, p-value: {}".format(anova_result.statistic, anova_result.pvalue))

if anova_result.pvalue < 0.05:
    print("There is a statistically significant difference in mean salaries across the education groups.")
else:
    print("There is no statistically significant difference in mean salaries across the education groups.")

# Checking assumptions for ANOVA
print("\n=== Checking Assumptions for ANOVA ===")

# Normality assumption using Kolmogorov-Smirnov test for large samples
print("Checking for normality using Kolmogorov-Smirnov test:")
stat_bachelors, p_bachelors = kstest(bachelors_salaries_cleaned, 'norm')
stat_masters, p_masters = kstest(masters_salaries_cleaned, 'norm')
stat_professional, p_professional = kstest(professional_salaries_cleaned, 'norm')

print(f"Bachelor's group: KS-statistic={stat_bachelors}, p-value={p_bachelors}")
print(f"Master's group: KS-statistic={stat_masters}, p-value={p_masters}")
print(f"Professional group: KS-statistic={stat_professional}, p-value={p_professional}")

if p_bachelors > 0.05 and p_masters > 0.05 and p_professional > 0.05:
    print("All groups follow a normal distribution (p > 0.05).")
else:
    print("At least one group does not follow a normal distribution (p <= 0.05).")

# Equal variance assumption using Levene's Test
print("\nChecking for equal variances using Levene's Test:")
stat_levene, p_levene = stats.levene(bachelors_salaries_cleaned, masters_salaries_cleaned, professional_salaries_cleaned)

print(f"Levene's test for equal variances: W={stat_levene}, p-value={p_levene}")

if p_levene > 0.05:
    print("Equal variance assumption holds (p > 0.05).")
else:
    print("Equal variance assumption does not hold (p <= 0.05).")


# %% [markdown]
# ## Bootstrapping and Mean Salary Differences by Education Level
# 
# This section applies **bootstrapping** to analyze salary differences between **Bachelor’s, Master’s, and Professional degrees**.
# 
# ### Key Steps:
# 1. **Filter Data** – Extract salary data for selected education levels.
# 2. **Remove Outliers** – Apply the **IQR method** for cleaner data.
# 3. **Bootstrap Sampling** – Generate 10,000 resampled means per education level.
# 4. **Compute Pairwise Salary Differences**:
#    - Bachelor’s vs. Master’s
#    - Bachelor’s vs. Professional
#    - Master’s vs. Professional
# 5. **Normalize Mean Differences** – Using **Welch’s t-score** for statistical comparison.
# 
# ### Results:
# - Provides **robust salary estimates** by reducing noise from small sample sizes.
# - Normalized mean differences help evaluate **statistical significance**.
# 
# This step strengthens **salary comparisons across education levels**.
# 

# %%


# Filter the data for the three education levels
bachelors_salaries = df[df['EdLevel'] == 'Bachelor’s degree']['ConvertedCompYearly']
masters_salaries = df[df['EdLevel'] == 'Master’s degree']['ConvertedCompYearly']
professional_salaries = df[df['EdLevel'] == 'Professional degree']['ConvertedCompYearly']

education_levels = ['Bachelor’s degree', 'Master’s degree', 'Professional degree']
df_filtered = df[df['EdLevel'].isin(education_levels)]


# SRemove outliers using the IQR method 
def remove_outliers_and_report(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    print(f"Number of outliers removed: {len(outliers)}")
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Remove outliers 
bachelors_salaries_cleaned = remove_outliers_and_report(bachelors_salaries)
masters_salaries_cleaned = remove_outliers_and_report(masters_salaries)
professional_salaries_cleaned = remove_outliers_and_report(professional_salaries)

# Define bootstrap parameters
n_replications = 10000

# Bootstrap sampling and mean calculation for each group
bootstrap_means = {}
for level in education_levels:
    salaries = df_filtered_cleaned[df_filtered_cleaned['EdLevel'] == level]['ConvertedCompYearly']
    bootstrap_means[level] = [
        np.mean(np.random.choice(salaries, size=len(salaries), replace=True)) for _ in range(n_replications)
    ]

# Calculate pairwise differences in means
diff_in_means_bachelor_master = np.array(bootstrap_means['Bachelor’s degree']) - np.array(bootstrap_means['Master’s degree'])
diff_in_means_bachelor_professional = np.array(bootstrap_means['Bachelor’s degree']) - np.array(bootstrap_means['Professional degree'])
diff_in_means_master_professional = np.array(bootstrap_means['Master’s degree']) - np.array(bootstrap_means['Professional degree'])

# Normalize the differences in means based on Welch's t-score
def normalize_diff(diff, group1, group2):
    var_group1 = np.var(group1, ddof=1)
    var_group2 = np.var(group2, ddof=1)
    n_group1 = len(group1)
    n_group2 = len(group2)
    se_diff = np.sqrt(var_group1 / n_group1 + var_group2 / n_group2)
    return diff / se_diff

norm_diff_bachelor_master = normalize_diff(diff_in_means_bachelor_master, bootstrap_means['Bachelor’s degree'], bootstrap_means['Master’s degree'])
norm_diff_bachelor_professional = normalize_diff(diff_in_means_bachelor_professional, bootstrap_means['Bachelor’s degree'], bootstrap_means['Professional degree'])
norm_diff_master_professional = normalize_diff(diff_in_means_master_professional, bootstrap_means['Master’s degree'], bootstrap_means['Professional degree'])


# %%
# Figure 1: Bootstrapped distributions for the three education levels
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_means['Bachelor’s degree'], color='blue', label="Bachelor's", kde=True, stat='count')
sns.histplot(bootstrap_means['Master’s degree'], color='green', label="Master's", kde=True, stat='count')
sns.histplot(bootstrap_means['Professional degree'], color='orange', label="Professional", kde=True, stat='count')
plt.title('Bootstrapped Distributions for Bachelor’s, Master’s, and Professional Degrees')
plt.xlabel('Bootstrapped Mean Salary (Annual Income)')
plt.ylabel('Count')
plt.legend()
plt.show()

# %%
# Figure 2: Distribution of pairwise differences in means (Bachelor vs Master, Bachelor vs Professional, Master vs Professional)
plt.figure(figsize=(10, 6))
sns.histplot(diff_in_means_bachelor_master, color='purple', label="Bachelor's - Master's", kde=True, stat='count')
sns.histplot(diff_in_means_bachelor_professional, color='red', label="Bachelor's - Professional", kde=True, stat='count')
sns.histplot(diff_in_means_master_professional, color='brown', label="Master's - Professional", kde=True, stat='count')
plt.title('Distribution of Pairwise Differences in Means (Bachelor, Master, Professional)')
plt.xlabel('Difference in Mean Salary')
plt.ylabel('Count')
plt.legend()
plt.show()

# %%
# Figure 3: Normalized distribution of differences in means
plt.figure(figsize=(10, 6))
sns.histplot(norm_diff_bachelor_master, color='purple', label="Normalized Bachelor - Master", kde=True, stat='count')
sns.histplot(norm_diff_bachelor_professional, color='red', label="Normalized Bachelor - Professional", kde=True, stat='count')
sns.histplot(norm_diff_master_professional, color='brown', label="Normalized Master - Professional", kde=True, stat='count')
plt.title('Normalized Distribution of Differences in Means')
plt.xlabel('Normalized Difference in Mean Salary')
plt.ylabel('Count')
plt.legend()
plt.show()

# %% [markdown]
# ## Bootstrapped ANOVA for Salary Differences
# 
# This section applies **bootstrapping and ANOVA** to test salary variations across education levels.
# 
# ### Key Steps:
# 1. **Bootstrap Sampling** – Generate 10,000 resampled salary means for:
#    - **Bachelor’s**
#    - **Master’s**
#    - **Professional degrees**
# 2. **Perform ANOVA on Bootstrapped Data** – Tests whether mean salaries differ significantly.
# 3. **Compare with Original ANOVA** – Validates bootstrapped results against the non-bootstrapped dataset.
# 
# ### Results:
# - If **p-value < 0.05**, at least one education group has a significantly different salary.
# - Bootstrapping ensures **robust statistical comparison**.
# 
# This step evaluates whether **education level significantly impacts salary distribution**.
# 

# %%

# Define bootstrap parameters
n_replications = 10000

# Bootstrap sampling and mean calculation for each group
bootstrap_means = {}
for level in education_levels:
    salaries = df_filtered_cleaned[df_filtered_cleaned['EdLevel'] == level]['ConvertedCompYearly']
    bootstrap_means[level] = [
        np.mean(np.random.choice(salaries, size=len(salaries), replace=True)) for _ in range(n_replications)
    ]

bachelors_bootstrap_means = np.array(bootstrap_means['Bachelor’s degree'])
masters_bootstrap_means = np.array(bootstrap_means['Master’s degree'])
prof_bootstrap_means = np.array(bootstrap_means['Professional degree'])


# Step 1: Perform ANOVA on the bootstrapped means
anova_stat_bootstrap, p_value_bootstrap = stats.f_oneway(
    bachelors_bootstrap_means, 
    masters_bootstrap_means, 
    prof_bootstrap_means
)

print(f"Bootstrap ANOVA F-statistic: {anova_stat_bootstrap}, p-value: {p_value_bootstrap}")

# Perform ANOVA on the original (non-bootstrapped) data
# First, filter the original salary data by education level
bachelors_salaries = df_filtered[df_filtered['EdLevel'] == "Bachelor’s degree"]['ConvertedCompYearly']
masters_salaries = df_filtered[df_filtered['EdLevel'] == "Master’s degree"]['ConvertedCompYearly']
prof_salaries = df_filtered[df_filtered['EdLevel'] == "Professional degree"]['ConvertedCompYearly']

anova_stat_original, p_value_original = stats.f_oneway(
    bachelors_salaries, 
    masters_salaries, 
    prof_salaries
)

print(f"Original ANOVA F-statistic: {anova_stat_original}, p-value: {p_value_original}")

# Step 3: Compare the results and explain the reasoning
if p_value_bootstrap < 0.05:
    print("The ANOVA test on bootstrapped data shows statistically significant differences between the groups.")
else:
    print("The ANOVA test on bootstrapped data does not show statistically significant differences between the groups.")

if p_value_original < 0.05:
    print("The ANOVA test on the original data shows statistically significant differences between the groups.")
else:
    print("The ANOVA test on the original data does not show statistically significant differences between the groups.")

# Visualize bootstrapped distributions
plt.figure(figsize=(10, 6))
sns.histplot(bachelors_bootstrap_means, color='blue', label="Bachelor's Degree", kde=True, stat="count")
sns.histplot(masters_bootstrap_means, color='green', label="Master's Degree", kde=True, stat="count")
sns.histplot(prof_bootstrap_means, color='red', label="Professional Degree", kde=True, stat="count")
plt.title("Bootstrapped Distributions of Salaries by Education Level")
plt.legend()
plt.show()



