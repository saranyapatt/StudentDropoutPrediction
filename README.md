# Student Dropout Prediction Project

This project focuses on identifying key predictors of student dropout and performing exploratory data analysis (EDA) on a dataset of 10,000 unique student records. The analysis explores demographics, academic performance, and student behavior to understand the factors contributing to educational attrition.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)

## Project Overview
The primary objective of this notebook is to process the "Student Dropout Prediction Dataset" to prepare it for machine learning applications. It includes data acquisition, initial statistical summaries, and identification of missing data patterns.

## Dataset Information
- **Source:** Kaggle (`meharshanali/student-dropout-prediction-dataset`).
- **Size:** 10,000 rows and 19 columns.
- **Target Variable:** `Dropout` (Binary: 0 = Stayed, 1 = Dropped out).

## Features
The dataset includes the following features categorized into three main groups:

### Personal Information
- `Student_ID`: Unique identifier.
- `Age`: Student age.
- `Gender`: Student gender.
- `Family_Income`: Total annual family income.
- `Internet_Access`: Presence of internet connectivity.
- `Parental_Education`: Highest educational attainment of parents.

### Academic Performance
- `GPA`: Overall Grade Point Average.
- `Semester_GPA`: GPA for the current semester.
- `CGPA`: Cumulative Grade Point Average.
- `Semester`: Current year of study (e.g., Year 1, Year 3).
- `Department`: Academic department (e.g., CS, Engineering, Arts, Business).

### Behavior & Lifestyle
- `Study_Hours_per_Day`: Average daily study time.
- `Attendance_Rate`: Percentage of class attendance.
- `Assignment_Delay_Days`: Average days for late assignment submissions.
- `Travel_Time_Minutes`: Commute time to the institution.
- `Part_Time_Job`: Employment status.
- `Scholarship`: Scholarship recipient status.
- `Stress_Index`: Measured level of student stress.

## Methodology
1. **Data Acquisition**: Automating dataset download using `kagglehub`.
2. **Exploratory Data Analysis**: 
   - Statistical profiling of numerical and categorical features.
   - Identifying missing values in key columns like `Family_Income`, `Study_Hours_per_Day`, and `Stress_Index`.
3. **Data Cleaning**: Preparing the dataset for predictive modeling by handling missing values and data types.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Kagglehub**: Direct dataset integration.

## Setup and Installation
To run this project, install the necessary Python libraries:

```bash
pip install kagglehub pandas matplotlib seaborn
