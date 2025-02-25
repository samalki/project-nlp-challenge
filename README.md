![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# PROJECT | Natural Language Processing Challenge

## Introduction

Learning how to process text is a skill required for Data Scientists/AI Engineers. 

In this project, you will put these skills into practice to identify whether a news headline is real or fake news.

## Project Overview

In the file `dataset/data.csv`, you will find a dataset containing news articles with the following columns:

- **`label`**: 0 if the news is fake, 1 if the news is real.
- **`title`**: The headline of the news article.
- **`text`**: The full content of the article.
- **`subject`**: The category or topic of the news.
- **`date`**: The publication date of the article.

Your goal is to build a classifier that is able to distinguish between the two.

Once you have a classifier built, then use it to predict the labels for `dataset/validation_data.csv`. Generate a new file
where the label `2` has been replaced by `0` (fake) or `1` (real) according to your model. Please respect the original file format, 
do not include extra columns, and respect the column separator. 

Please ensure to split the `data.csv` into **training** and **test** datasets before using it for model training or evaluation.

## Guidance

Like in a real life scenario, you are able to make your own choices and text treatment.
Use the techniques you have learned and the common packages to process this data and classify the text.

## Deliverables

1. **Python Code:** Provide well-documented Python code that conducts the analysis.
2. **Predictions:** A csv file in the same format as `validation_data.csv` but with the predicted labels (0 or 1)
3. **Accuracy estimation:** Provide the teacher with your estimation of how your model will perform.
4. **Presentation:** You will present your model in a 10-minute presentation. Your teacher will provide further instructions.
