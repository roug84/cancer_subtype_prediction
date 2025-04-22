# init-minio.py
import boto3
import os
from botocore.client import Config

def create_bucket():
    # Initialize MinIO client
    # This creates a client that will connect to MinIO running in Docker
    s3_client = boto3.client(
        's3',
        # This line specifically connects to the MinIO container through port 9000
        # 'localhost' refers to your host machine, and Docker forwards this to the container
        endpoint_url='http://localhost:9002',
        # These credentials match what's in your docker-compose.yml for MinIO
        aws_access_key_id='kensou',
        aws_secret_access_key='H03042020P16082022',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

    # This line creates a bucket in the MinIO running in Docker
    try:
        s3_client.create_bucket(Bucket='mlflow')
        print("Successfully created 'mlflow' bucket")
    except s3_client.exceptions.BucketAlreadyExists:
        print("Bucket 'mlflow' already exists")
    except Exception as e:
        print(f"Error creating bucket: {str(e)}")

if __name__ == "__main__":
    create_bucket()