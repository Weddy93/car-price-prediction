# Used-Car Price Prediction

A machine-learning regression project that estimates the selling price
of a used car using vehicle characteristics. This project was developed using Python and scikit-learn in
Google Colab.

> **Project status:** Model training, evaluation, validation, and
> artifact saving are complete. This is an internship/research project
> and is not currently deployed to production.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Dataset](#dataset)
-   [Technology Stack](#technology-stack)
-   [Workflow](#workflow)
-   [Features Used](#features-used)
-   [Models Evaluated](#models-evaluated)
-   [Final Model Results](#final-model-results)
-   [Error Analysis and Limitations](#error-analysis-and-limitations)
-   [Saved Artifacts](#saved-artifacts)
-   [Run the Project](#run-the-project)
-   [Future Improvements](#future-improvements)

## Project Overview

The goal is to predict the selling price of a used car from information
such as its brand, model, age, mileage, fuel type, seller type,
transmission, and ownership history.

The project uses a dataset of used-car listings from **India**, and
prices are represented in **Indian rupees (INR, ₹)**. Results should
therefore be interpreted in the context of this dataset and should not
be treated as estimates for the Kenyan market.

## Objectives

-   Explore and clean the used-car dataset.
-   Identify patterns and relationships that may affect selling prices.
-   Engineer useful features from vehicle names and years.
-   Train and compare regression models.
-   Tune a Random Forest model and evaluate it using a held-out test set
    and cross-validation.
-   Analyze prediction errors and save the fitted
    preprocessing-and-model pipeline.

## Dataset

The working dataset originally contained **4,340 rows and 8 columns**.
After removing **763 exact duplicate rows**, the cleaned dataset
contained **3,577 rows**.

The source columns were:

  Column            Description
  ----------------- -----------------------------------------
  `name`            Vehicle name as listed in the dataset
  `year`            Vehicle manufacturing year
  `selling_price`   Selling price in INR; prediction target
  `km_driven`       Recorded distance driven in kilometres
  `fuel`            Fuel category
  `seller_type`     Seller category
  `transmission`    Transmission category
  `owner`           Ownership-history category

The project created additional features, including `brand`, `car_age`,
and `car_model`.

**Data note:** The reference year used to calculate vehicle age is 2026.
The dataset has known limitations and some records require source-level
verification. The original dataset source and licence should be recorded
here before public redistribution.

## Technology Stack

-   Python
-   pandas and NumPy
-   scikit-learn
-   Matplotlib and Seaborn for exploratory visualisation
-   Joblib for saving the fitted model pipeline
-   Google Colab / Jupyter Notebook

## Workflow

1.  **Data inspection:** Reviewed column types, missing values,
    duplicates, value ranges, and category counts.
2.  **Data cleaning:** Removed exact duplicate rows and checked basic
    data-quality conditions.
3.  **Exploratory data analysis (EDA):** Examined price distributions
    and relationships between selling price and vehicle characteristics.
4.  **Feature engineering:** Extracted brand and a simplified model
    label from the vehicle name, and calculated vehicle age using
    `2026 - year`.
5.  **Train/test split:** Used an 80/20 split with `random_state=42`,
    resulting in 2,861 training rows and 716 test rows.
6.  **Preprocessing:** Applied one-hot encoding to categorical features
    with unknown categories ignored at prediction time, while passing
    numerical features through.
7.  **Model comparison:** Evaluated Linear Regression, Decision Tree,
    and Random Forest regressors.
8.  **Hyperparameter tuning:** Used `RandomizedSearchCV` with five-fold
    cross-validation and negative mean absolute error as the selection
    score.
9.  **Final evaluation:** Reported MAE, RMSE, and R² on the held-out
    test set and examined large prediction errors.
10. **Artifact saving:** Saved the complete fitted pipeline and a JSON
    metadata file.

## Features Used

The final Version 2 model uses these eight input features:

  ---------------------------------------------------------------------
  Feature                            Description
  ---------------------------------- ----------------------------------
  `brand`                            Vehicle brand extracted from the
                                     name

  `car_model`                        Simplified model label extracted
                                     from the vehicle name

  `car_age`                          Vehicle age in years, calculated
                                     using the 2026 reference year

  `km_driven`                        Recorded distance driven in
                                     kilometres

  `fuel`                             Fuel category

  `seller_type`                      Seller category

  `transmission`                     Transmission category

  `owner`                            Ownership-history category
  ---------------------------------------------------------------------

Categorical features are one-hot encoded within the saved scikit-learn
pipeline. Keeping preprocessing inside the pipeline helps ensure that
prediction-time inputs are processed consistently with training data.

## Models Evaluated

Initial held-out test-set comparison:

  Model                    MAE (₹)     RMSE (₹)       R²
  ------------------- ------------ ------------ --------
  Linear Regression     206,368.40   431,734.10   0.4214
  Decision Tree         199,738.94   475,330.13   0.2986
  Random Forest         155,993.69   358,572.56   0.6009

The Random Forest model was selected for further tuning. The final
candidate uses 400 trees, maximum depth 15, `min_samples_split=10`,
`min_samples_leaf=1`, and `max_features=0.7`.

An experiment with a revised rule-based model-name extractor did not
improve the evaluation metrics, so Version 2 was retained as the
preferred candidate.

## Final Model Results

**Selected candidate:** Version 2 --- Random Forest with `car_model`

  Metric     Held-out test set   Five-fold cross-validation mean
  -------- ------------------- ---------------------------------
  MAE              ₹127,157.37                       ₹116,226.31
  RMSE             ₹355,136.92                       ₹266,804.74
  R²                    0.6085                            0.7020

Cross-validation was run on the training portion of the data. The final
test set contains 716 rows and was used for the held-out evaluation.

### What the metrics mean

-   **MAE (Mean Absolute Error):** The average absolute difference
    between actual and predicted prices. The held-out MAE is
    approximately ₹127,157.
-   **RMSE (Root Mean Squared Error):** A prediction-error measure that
    penalises larger errors more heavily than MAE.
-   **R²:** Measures how much of the variation in the target is
    explained by the model on the evaluated data. A test R² of 0.6085
    does **not** mean the model is "60.85% accurate."

The model's mean absolute error is useful as a broad evaluation measure,
but individual predictions can differ substantially from actual prices.

## Error Analysis and Limitations

Error analysis showed that predictions were generally more difficult for
expensive and less frequently represented vehicles. Some premium
Mercedes-Benz and BMW examples were substantially underpredicted, while
some other cars were overpredicted.

The test-set error review also identified records that deserve
verification against the original listings, including a Toyota Camry
Hybrid with an `Electric` fuel label and a Tata Harrier XE recorded at
₹426,000 with 1,000 km driven. These are **suspected data-quality
concerns, not confirmed errors**, and were not automatically removed.

Known limitations:

-   The dataset represents Indian listings and uses INR; it is not a
    validated Kenyan-market pricing model.
-   The model-name extraction is rule-based and does not fully capture
    every model, trim, engine, or variant.
-   Rare and expensive cars have limited comparable examples, which can
    produce large errors.
-   Some source records may contain inaccurate, unusual, or inconsistent
    values.
-   The reference year for `car_age` is fixed at 2026; it should be
    updated deliberately if the model is retrained for a later period.
-   Performance metrics describe this dataset and split; results may
    differ on new data.

## Saved Artifacts

The model and metadata were saved in Google Colab at:

``` text
/content/model_artifacts/used_car_price_model_v2.joblib
/content/model_artifacts/used_car_price_model_v2_metadata.json
```

-   `used_car_price_model_v2.joblib` --- the fitted scikit-learn
    pipeline, including preprocessing and the Random Forest regressor.
-   `used_car_price_model_v2_metadata.json` --- model description,
    feature schema, data split sizes, evaluation metrics, and limitation
    notes.

These files are stored in the Colab runtime unless downloaded or copied
to persistent storage. They are not automatically included in a Git
repository.

**Security note:** Only load Joblib files from trusted sources because
loading serialized Python objects can execute code.

## Run the Project

1.  Open the project notebook in Google Colab.
2.  Upload the dataset used by the notebook, ensuring it is the expected
    version and that you have permission to use it.
3.  Run the notebook cells in order, from data inspection through model
    evaluation and saving.
4.  To use the saved pipeline in a Python environment, make sure
    compatible versions of Python, scikit-learn, pandas, NumPy, and
    Joblib are installed.
5.  Load the saved model and pass a pandas DataFrame containing all
    eight required features with the expected column names.

Example of loading the saved model:

``` python
import joblib
import pandas as pd

model = joblib.load("used_car_price_model_v2.joblib")

sample_car = pd.DataFrame([{
    "brand": "Maruti",
    "car_model": "Swift",
    "car_age": 5,
    "km_driven": 45000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner"
}])

predicted_price = model.predict(sample_car)[0]
print(f"Estimated selling price: ₹{predicted_price:,.2f}")
```

The example values are illustrative and are not a claim about the
current market price of that vehicle. The model expects feature values
in the same general format as the training data. This project has not
been deployed to production.

## Future Improvements

-   Verify questionable records against the original source.
-   Improve model and trim extraction, potentially using a structured
    vehicle catalogue.
-   Evaluate additional models and price transformations.
-   Report performance separately for price bands, brands, and common
    versus rare models.
-   Add automated input validation and tests.
-   Record exact dependency versions and the dataset source/licence for
    reproducibility.
-   Obtain representative Kenyan used-car data and retrain/evaluate
    separately if the intended use becomes Kenyan-market pricing.

## Acknowledgement

Developed as a used-car price prediction machine-learning project for an
OasisByte internship task.
