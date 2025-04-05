import os
import mlflow
import logging
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def promote_model():
    """Promote the latest staging model to production with proper validation"""
    try:
        # Configure MLflow
        mlflow.set_tracking_uri("http://ec2-3-110-174-70.ap-south-1.compute.amazonaws.com:5000/")
        client = MlflowClient()
        model_name = "yt_chrome_plugin_model"

        # Get latest staging version using modern API
        staging_versions = client.search_model_versions(
            filter_string=f"name='{model_name}' AND tag.stage='Staging'",
            order_by=["last_updated_timestamp DESC"],
            max_results=1
        )

        if not staging_versions:
            raise ValueError(f"No staging model found for {model_name}")
            
        staging_version = staging_versions[0]
        logger.info(f"Found staging model version: {staging_version.version}")

        # Archive current production versions
        prod_versions = client.search_model_versions(
            filter_string=f"name='{model_name}' AND tag.stage='Production'"
        )

        if prod_versions:
            logger.info(f"Archiving {len(prod_versions)} production versions")
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived",
                    archive_existing_versions=True
                )

        # Promote new version to production
        client.set_model_version_tag(
            name=model_name,
            version=staging_version.version,
            key="stage",
            value="Production"
        )

        # Verify promotion
        new_prod_version = client.get_model_version(
            name=model_name,
            version=staging_version.version
        )

        if new_prod_version.current_stage != "Production":
            raise RuntimeError("Model promotion failed")

        logger.info(f"Successfully promoted version {staging_version.version} to Production")
        return True

    except MlflowException as e:
        logger.error(f"MLflow operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if promote_model():
        logger.info("Model promotion completed successfully")
    else:
        logger.error("Model promotion failed")
        exit(1)