"""
Run the full Feast MLOps pipeline end-to-end.

This replaces re-running feast.ipynb top-to-bottom. Each step lives in its
own module so you can also run them individually. Steps assume the source
table `airline_features.parquet` already exists (produced by setup_feast.py).

    python run_pipeline.py
"""
import split_features
import apply_features
import create_training_dataset
import rank_features
import train_top_features
import materialize_online
import predict_online

STEPS = [
    ("Split features into predictor/target", split_features.main),
    ("Apply feature definitions (feast apply)", apply_features.main),
    ("Build saved training dataset", create_training_dataset.main),
    ("Rank features by importance", rank_features.main),
    ("Train models on top features", train_top_features.main),
    ("Materialize online store", materialize_online.main),
    ("Predict from online store", predict_online.main),
]


def main():
    for i, (title, step) in enumerate(STEPS, 1):
        print("\n" + "=" * 70)
        print(f"STEP {i}/{len(STEPS)}: {title}")
        print("=" * 70)
        step()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
