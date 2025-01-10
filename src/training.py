import mlflow
from mlflow import MlflowClient
from ultralytics import YOLO

def train_coco128():
    with mlflow.start_run(run_name="yolov11n", log_system_metrics=True) as run:
        model = YOLO("yolov10n.pt")
        # model.train(data='coco128.yaml', epochs=1, save_dir='../')
        mlflow.log_artifact(local_path="requirements.txt", artifact_path="environment", run_id=run.info.run_id)

def register_model():
    run_id = mlflow.last_active_run().info.run_id
    model_name = "coco_yolov10n"

    model_version = mlflow.register_model(model_uri=f"runs:/{run_id}/weights/best.pt", name=model_name)

    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias="Champion", version=model_version.version)

if __name__ == '__main__':
    train_coco128()
    register_model()