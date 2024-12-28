import mlflow

run_id = 'e19f0b914fbb48ee817d1b40a686d9b7'
download_path = './download'

model_uri = f'runs:/{run_id}/model'

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_path)
# use pickle code