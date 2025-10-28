import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A simple group-based estimator that predicts a numeric value
    based on categorical group means or medians.
    """

    def __init__(self, estimate="mean"):
        """
        Initialize the estimator

        Parameters
        ----------
        estimate : str
            Either mean or median, and determines which aggregation
            method will be used during fitting
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either mean or median")

        self.estimate = estimate
        self.group_estimates = None
        self.default_category = None
        self.default_estimates = None

    def fit(self, X, y, default_category=None):
        """
        Fit the estimator with categorical features X and numeric target y.

        Parameters
        ----------
        X :  Categorical features to group by.
        y : Continuous values corresponding to X.
        default_category : str, optional
            A single column name in X to use as a fallback if a full
            category combination is missing during prediction.
        """
        # validation
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame of categorical columns.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if pd.isna(pd.Series(y)).any():
            raise ValueError("y contains missing values; remove or impute them before fitting.")

        self.default_category = default_category

        # Combine features and target into one DataFrame
        df = pd.concat([X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1)
        target_name = y.name if hasattr(y, "name") and y.name is not None else df.columns[-1]

        # Compute group-level mean or median
        if self.estimate == "mean":
            group_est = df.groupby(list(X.columns), observed=True)[target_name].mean()
        else:
            group_est = df.groupby(list(X.columns), observed=True)[target_name].median()

        self.group_estimates = group_est

        # Compute fallback estimates if default_category is provided and present
        self.default_estimates = None
        if default_category is not None:
            if default_category not in X.columns:
                # keep default_estimates = None but warns user
                print(
                    f"Warning: default_category='{default_category}' not in X columns; "
                    "fallback will not be available."
                )
            else:
                if self.estimate == "mean":
                    self.default_estimates = df.groupby(default_category, observed=True)[
                        target_name
                    ].mean()
                else:
                    self.default_estimates = df.groupby(default_category, observed=True)[
                        target_name
                    ].median()

    def predict(self, X_):
        """
        Predict values for new categorical observations.

        Parameters
        ----------
        X_ : Observations containing the same columns as X used in fit().

        Returns
        -------
        np.ndarray
            Predicted mean or median values for each observation.
            If a group is missing, np.nan is returned for that case.
        """
        # Ensure we have fit() before predicting
        if self.group_estimates is None:
            raise RuntimeError("Model has not been fitted. Call fit(X, y) before predict().")

        if isinstance(X_, pd.DataFrame):
            X_df = X_.copy()
        else:
            X_df = pd.DataFrame(X_, columns=self.group_estimates.index.names)

        expected_cols = list(self.group_estimates.index.names)
        if list(X_df.columns) != expected_cols:
            # reorder if they contain same names in different order
            if set(X_df.columns) == set(expected_cols):
                X_df = X_df[expected_cols]
            else:
                raise ValueError(
                    f"Prediction input columns {list(X_df.columns)} do not match expected "
                    f"{expected_cols}."
                )

        results = []
        missing_count = 0

        for _, row in X_df.iterrows():
            key = tuple(row.values)

            # Exact match for a known group
            if key in self.group_estimates.index:
                results.append(self.group_estimates.loc[key])

            # Fallback: use single-category estimate if available
            elif self.default_category is not None and self.default_category in X_df.columns:
                cat_value = row[self.default_category]
                if self.default_estimates is not None and cat_value in self.default_estimates.index:
                    results.append(self.default_estimates.loc[cat_value])
                else:
                    results.append(np.nan)
                    missing_count += 1

            # If no match and no fallback
            else:
                results.append(np.nan)
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} group(s) were missing; returned NaN for those.")

        # Convert results to array
        return np.array([float(r) if not pd.isna(r) else np.nan for r in results])