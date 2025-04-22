import os
from minio import Minio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_model_files():
    """Download required model files from MinIO on container startup"""
    try:
        # Connect to production MinIO server
        minio_client = Minio(
            os.getenv('MINIO_ENDPOINT', '77.205.31.96:9000'),
            access_key=os.getenv('MINIO_ACCESS_KEY', 'kensou'),
            secret_key=os.getenv('MINIO_SECRET_KEY', 'H03042020P16082022'),
            secure=False  # Set to True if using HTTPS
        )

        bucket_name = "cancer-subtype"  # Your bucket name
        model_path = 'TCGA_BRCA_vminio_postgre_2'

        # Ensure directory exists
        os.makedirs(f'/app/results/tcga/{model_path}', exist_ok=True)

        # List of files to download
        files_to_download = [
            'shap_values.pkl',
            # Add other required files here
        ]

        for file_name in files_to_download:
            local_path = f'/app/results/tcga/{model_path}/{file_name}'

            logger.info(f"Downloading {file_name} from MinIO...")
            minio_client.fget_object(
                bucket_name,
                f'{model_path}/{file_name}',
                local_path
            )
            logger.info(f"Successfully downloaded {file_name}")

    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        raise


if __name__ == "__main__":
    download_model_files()