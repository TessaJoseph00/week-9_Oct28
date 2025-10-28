import pandas as pd
from apputil import GroupEstimate


def main():
    """Run group-based estimation on the coffee dataset."""

    df = pd.read_csv("coffee_analysis.csv")

    # Select predictors (categorical features) and target (continuous variable)
    X = df[["loc_country", "roast"]]
    y = df["rating"]

    # Initialize and fit model using mean as the aggregation
    gm = GroupEstimate(estimate="mean")
    gm.fit(X, y)

    # Example input data for prediction
    X_new = [
        ["United States", "Medium-Light"],
        ["Hong Kong", "Medium-Light"],
        ["Brazil", "Dark"]
    ]

    # Display predictions
    print("Predictions:", gm.predict(X_new))

    # Example using a fallback category for missing combinations
    gm2 = GroupEstimate(estimate="mean")
    gm2.fit(X, y, default_category="loc_country")

    print("Predictions with fallback:", gm2.predict(X_new))


if __name__ == "__main__":
    main()
