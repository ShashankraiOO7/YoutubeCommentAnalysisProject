import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-3-110-174-70.ap-south-1.compute.amazonaws.com:5000/")
    client = MlflowClient()
    model_name = "yt_chrome_plugin_model"

    # Get the latest version in Staging (using the old stage, if still set)
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise ValueError(f"No model found in the 'Staging' stage for '{model_name}'")
    latest_version_staging = staging_versions[0].version

    # Attempt to fetch the current production model by alias ("champion")
    try:
        # This method retrieves the model version associated with the alias "champion"
        production_model = client.get_model_version_by_alias(model_name, "champion")
        # Mark the current production model as archived using a tag
        client.set_model_version_tag(model_name, production_model.version, "status", "archived")
        # Remove the "champion" alias from the current production model
        client.delete_model_version_alias(model_name, "champion")
        print(f"Archived previous production model version {production_model.version}")
    except Exception as e:
        # If no production model with alias "champion" is found, we continue
        print("No existing production model with alias 'champion' found. Proceeding with promotion.")

    # Promote the new model by assigning the alias "champion" to the latest staging version
    client.set_model_version_alias(model_name, "champion", latest_version_staging)
    print(f"Model version {latest_version_staging} promoted as 'champion' (Production)")

if __name__ == "__main__":
    promote_model()
