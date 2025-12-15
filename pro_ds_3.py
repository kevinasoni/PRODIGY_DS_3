import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Load dataset
    data_path = "data/bank-additional-full.csv"
    df = pd.read_csv(data_path, sep=";")

    # Separate features and target
    X = df.drop("y", axis=1)
    y = df["y"].map({"yes": 1, "no": 0})

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    # Decision Tree model
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=50,
        random_state=42
    )

    # Build pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Extract trained decision tree for visualization
    tree_model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    # Output results
    print("Decision Tree Classifier Results")
    print("-------------------------------")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("
Classification Report:")
    print(class_report)

    # Save decision tree visualization
    try:
        from sklearn.tree import export_graphviz
        import graphviz

        dot_data = export_graphviz(
            tree_model,
            out_file=None,
            feature_names=feature_names,
            class_names=["No Purchase", "Purchase"],
            filled=True,
            rounded=True,
            special_characters=True
        )

        graph = graphviz.Source(dot_data)
        graph.render("results/decision_tree_visualization", format="png", cleanup=True)
        print("
Decision tree visualization saved to results/decision_tree_visualization.png")

    except ImportError:
        print("
Graphviz is not installed. Install graphviz to enable tree visualization.")


if __name__ == "__main__":
    main()
