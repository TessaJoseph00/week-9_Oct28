import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A simple group-based estimator that predicts a numeric value
    based on categorical group means or medians.
    """

    def __init__(self, estimate="mean"):
        """
        Initialize the estimator.

        Parameters
        ----------
        estimate : str
            Either 'mean' or 'median'. Determines which aggregation
            method will be used during fitting.
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None
        self.default_category = None
        self.default_estimates = None

    def fit(self, X, y, default_category=None):
        """
        Fit the estimator with categorical features X and numeric target y.

        Parameters
        ----------
        X : Categorical features to group by.
        y : Continuous values corresponding to X.
        default_category : str, optional
            A single column name in X to use as a fallback if a full
            category combination is missing during prediction.
        """
        self.default_category = default_category

        # Combine features and target into one DataFrame
        df = pd.concat([X, y], axis=1)
        target_name = y.name if hasattr(y, "name") else "target"

        # Calculate group-level mean or median
        if self.estimate == "mean":
            group_est = df.groupby(list(X.columns), observed=True)[target_name].mean()
        else:
            group_est = df.groupby(list(X.columns), observed=True)[target_name].median()

        self.group_estimates = group_est

        # Compute fallback estimates if default_category is provided
        if default_category and default_category in X.columns:
            if self.estimate == "mean":
                self.default_estimates = (
                    df.groupby(default_category, observed=True)[target_name].mean()
                )
            else:
                self.default_estimates = (
                    df.groupby(default_category, observed=True)[target_name].median()
                )

    def predict(self, X_):
        """
        Predict values for new categorical observations.

        Parameters
        ----------
        X_ : list or pd.DataFrame
            Observations containing the same columns as X used in fit().

        Returns
        -------
        np.ndarray
            Predicted mean or median values for each observation.
            If a group is missing, np.nan is returned for that case.
        """
        X_ = pd.DataFrame(X_, columns=self.group_estimates.index.names)
        results = []
        missing_count = 0

        for _, row in X_.iterrows():
            key = tuple(row.values)

            # Exact match for a known group
            if key in self.group_estimates.index:
                results.append(self.group_estimates.loc[key])

            # Fallback: use single-category estimate if available
            elif self.default_category and self.default_category in X_.columns:
                cat_value = row[self.default_category]
                if (
                    self.default_estimates is not None
                    and cat_value in self.default_estimates.index
                ):
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

        # Return as plain floats (not np.float64)
        return np.array([float(r) if not pd.isna(r) else np.nan for r in results])

