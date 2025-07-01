from pipelines.pipeline import training_pipeline


if __name__ == "__main__":
    try:
        training_pipeline()
    except Exception as e:
        raise e
