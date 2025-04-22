import mlflow
import os
import boto3
from botocore.client import Config


def create_mlflow_bucket():
    """Create MLflow bucket in MinIO if it doesn't exist"""
    try:
        print("Creating MinIO bucket...")
        s3_client = boto3.client(
            's3',
            endpoint_url='http://15.188.85.249:9002',
            aws_access_key_id='kensou',
            aws_secret_access_key='H03042020P16082022',
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )

        # Create bucket if it doesn't exist
        try:
            s3_client.create_bucket(Bucket='mlflow')
            print("Successfully created 'mlflow' bucket")
        except s3_client.exceptions.BucketAlreadyExists:
            print("Bucket 'mlflow' already exists")
        except Exception as e:
            print(f"Error creating bucket: {str(e)}")
            raise
    except Exception as e:
        print(f"Error connecting to MinIO: {str(e)}")
        raise


def copy_model_to_docker_mlflow():
    """Copy models from local MLflow to containerized MLflow"""
    try:
        print("Starting model copy process...")

        # First ensure the bucket exists
        create_mlflow_bucket()

        # Connect to local MLflow and set credentials for local
        print("Connecting to local MLflow...")
        mlflow.set_tracking_uri("http://localhost:8000")

        # Set local MinIO credentials
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
        os.environ['AWS_ACCESS_KEY_ID'] = "kensou"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "H03042020P16082022"

        # Download models
        local_models = []
        for i in range(1, 11):
            model_name = f"cancer_subtype_predictor_ensemble_{i}"
            print(f"Downloading local model {model_name}...")

            local_model = mlflow.sklearn.load_model(
                model_uri=f"models:/{model_name}/Production"
            )
            local_models.append((model_name, local_model))

        # Switch to EC2 MLflow configuration
        print("Switching to EC2 MLflow...")
        mlflow.set_tracking_uri("http://15.188.85.249:8001")

        # Set MinIO credentials for EC2
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://15.188.85.249:9002"
        os.environ['AWS_ACCESS_KEY_ID'] = "kensou"
        os.environ['AWS_SECRET_ACCESS_KEY'] = "H03042020P16082022"

        # Log each model to Docker MLflow
        client = mlflow.tracking.MlflowClient()

        for model_name, local_model in local_models:
            print(f"Copying {model_name} to Docker MLflow...")

            with mlflow.start_run() as run:
                mlflow.sklearn.log_model(
                    sk_model=local_model,
                    artifact_path="model",
                    registered_model_name=model_name
                )

                # Get latest version and set to production
                versions = client.get_latest_versions(model_name)
                if versions:
                    latest_version = versions[0].version
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage="Production"
                    )
                    print(f"Set {model_name} version {latest_version} to Production")

        print("Model copy process completed successfully!")

    except Exception as e:
        print(f"Error during model copy: {str(e)}")
        raise


if __name__ == "__main__":
    copy_model_to_docker_mlflow()