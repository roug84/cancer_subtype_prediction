# upload_to_minio.py
from minio import Minio
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_directory_to_minio(minio_client, bucket_name, local_dir, minio_path_prefix):
    """Recursively upload a directory and its contents to MinIO"""
    for root, dirs, files in os.walk(local_dir):
        for file_name in files:
            # Get the full local path
            local_file_path = os.path.join(root, file_name)

            # Calculate the MinIO path
            relative_path = os.path.relpath(local_file_path, local_dir)
            minio_path = f"{minio_path_prefix}/{relative_path}"

            try:
                logger.info(f"Uploading {relative_path}...")
                minio_client.fput_object(
                    bucket_name,
                    minio_path,
                    local_file_path
                )
                logger.info(f"Successfully uploaded {relative_path}")
            except Exception as e:
                logger.error(f"Failed to upload {relative_path}: {str(e)}")


def upload_to_minio():
    logger.info("Starting MinIO upload process...")

    try:
        logger.info("Connecting to MinIO server...")
        minio_client = Minio(
            "77.205.31.96:9000",
            access_key="kensou",
            secret_key="H03042020P16082022",
            secure=False
        )
        logger.info("Successfully connected to MinIO server")

        bucket_name = "cancer-subtype"
        if not minio_client.bucket_exists(bucket_name):
            logger.info(f"Creating bucket: {bucket_name}")
            minio_client.make_bucket(bucket_name)
            logger.info(f"Successfully created bucket: {bucket_name}")
        else:
            logger.info(f"Bucket {bucket_name} already exists")

        # Set the local directory path
        local_path = "/home/hector/roug/cancer_subtype_prediction/results/tcga/TCGA_BRCA_vminio_postgre_2"
        if not os.path.exists(local_path):
            logger.error(f"Local path does not exist: {local_path}")
            return

        # Upload the entire directory structure
        upload_directory_to_minio(
            minio_client,
            bucket_name,
            local_path,
            "TCGA_BRCA_vminio_postgre_2"
        )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    upload_to_minio()