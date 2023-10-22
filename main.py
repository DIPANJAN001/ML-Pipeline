from ml_main_pipeline import MLMainPipeline

if __name__ == "__main__":
    main_pipeline = MLMainPipeline()
    accuracy, report = main_pipeline.run_pipeline()

    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", report)
