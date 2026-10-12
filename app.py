from flask import Flask, render_template

from models.load import load_dataset
from models.eda import (
    dataset_info,
    dataset_description,
    missing_values,
    duplicate_values,
    glucose_statistics,
    glucose_distribution,
    glucose_boxplot,
    correlation_heatmap,
)
from models.preprocessing import missing_report, prepare_features
from models.regression import train_regression


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load")
def load():
    df = load_dataset()
    return render_template(
        "load.html",
        data=df.head().to_html(classes="data-table", index=False),
        rows=df.shape[0],
        columns=df.shape[1],
    )


@app.route("/eda")
def eda():
    df = load_dataset()
    description = dataset_description(df)
    missing = missing_values(df)
    correlation, heatmap_image = correlation_heatmap(df)

    return render_template(
        "eda.html",
        info=dataset_info(df),
        description=description.to_html(classes="data-table"),
        missing=missing[missing > 0].to_frame("Missing").to_html(
            classes="data-table"
        ),
        duplicates=duplicate_values(df),
        glucose_stats=glucose_statistics(df),
        distribution_image=glucose_distribution(df),
        boxplot_image=glucose_boxplot(df),
        correlation=correlation.to_html(classes="data-table", float_format=lambda x: f"{x:.2f}"),
        heatmap_image=heatmap_image,
    )


@app.route("/preprocessing")
def preprocessing():
    df = load_dataset()
    report = missing_report(df)
    X, y, _, numeric, categorical = prepare_features(df)

    before_missing = int(df.isna().sum().sum())
    after_knn = 0  # KNN is fitted inside the training pipeline; no leakage.

    return render_template(
        "preprocessing.html",
        rows=len(df),
        columns=len(df.columns),
        before_missing=before_missing,
        report=report.to_html(classes="data-table"),
        numeric_count=len(numeric),
        categorical_count=len(categorical),
        model_features=X.shape[1],
        after_knn=after_knn,
    )


@app.route("/regression")
def regression():
    metrics, results, model_path, plot_image = train_regression()
    return render_template(
        "regression.html",
        metrics=metrics,
        results=results.head(15).to_html(classes="data-table", index=False),
        model_path=model_path,
        plot_image=plot_image,
    )


if __name__ == "__main__":
    # Run Flask only; EDA plots are generated when /eda is opened.
    app.run(debug=True, use_reloader=False)
