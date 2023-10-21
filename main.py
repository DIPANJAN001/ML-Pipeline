from ml_main_pipeline import MLMainPipeline
import streamlit as st
if __name__ == "__main__":
    main_pipeline = MLMainPipeline()
    accuracy, report = main_pipeline.run_pipeline()
    st.write(f"Accuracy: {accuracy} %")
    # print("Model Accuracy:", accuracy)
    # print("Classification Report:\n", report)
