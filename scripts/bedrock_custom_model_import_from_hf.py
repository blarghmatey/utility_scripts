from datetime import date
from pathlib import Path
import shutil
import boto3
import os
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Configuration
huggingface_model_id = os.environ.get(
    "HF_MODEL_REPO", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)
aws_region = "us-east-1"  # Replace with your desired AWS region
bedrock_model_name = os.environ.get(
    "BEDROCK_MODEL_NAME", f"{huggingface_model_id.split('/')[-1]}-{date.today()}"
)  # Choose a unique name for your Bedrock model
hf_token = os.environ.get("HF_TOKEN")  # Your Hugging Face API token
s3_bucket_name = os.environ.get("S3_BUCKET_NAME")  # Your S3 bucket name

# Initialize AWS client for Bedrock
bedrock_client = boto3.client("bedrock", region_name=aws_region)
s3_client = boto3.client("s3")


# Function to download the Hugging Face model to local storage
def download_huggingface_model():
    target_path = Path('hf_model')
    target_path.mkdir(exist_ok=True)
    model_path = snapshot_download(
        repo_id=huggingface_model_id, token=hf_token, local_dir=target_path
    )
    print(f"Model downloaded to {model_path}")
    return Path(model_path)


def upload_hf_model_to_s3(model_path: Path, bucket_prefix: str | None = None):
    """Upload the downloaded model files to an S3 bucket for use by AWS Bedrock."""
    if not bucket_prefix:
        bucket_prefix = bedrock_model_name
    for file_path in model_path.iterdir():
        if file_path.is_dir():
            continue
        s3_key = file_path.relative_to(model_path)
        print(f"Preparing to upload {file_path} to s3://{s3_bucket_name}/{bucket_prefix}/ with key {s3_key}")
        file_size = file_path.stat().st_size

        with tqdm(total=file_size, desc=str(file_path), unit="B", unit_scale=True) as pbar:
            s3_client.upload_file(
                file_path, s3_bucket_name, f"{bucket_prefix}/{s3_key}", Callback=pbar.update
            )
        print(
            f"Uploaded {file_path} to s3://{s3_bucket_name}/{bucket_prefix}/{s3_key}"
        )
    shutil.rmtree(model_path)
    return f"s3://{s3_bucket_name}/{bucket_prefix}"


# Function to create a Bedrock model using the Hugging Face model
def createbedrockmodel(s3_uri: str, model_name: str):
    """Create a Bedrock model using the Hugging Face model files uploaded to S3."""
    return bedrock_client.create_model_import_job(
        jobName=f"import-{model_name}",
        importedModelName=model_name,
        roleArn="arn:aws:iam::610119931565:role/service-role/bedrock-custom-model-s3-import-role",
        modelDataSource={"s3DataSource": {"s3Uri": s3_uri}},
        clientRequestToken=bedrock_model_name,
    )


def main():
    # Download the Hugging Face model
    model_path = download_huggingface_model()
    s3_uri = upload_hf_model_to_s3(model_path)
    br_model = createbedrockmodel(s3_uri, bedrock_model_name)
    print("Bedrock Model Creation Response:", br_model)


if __name__ == "__main__":
    print("Importing ", bedrock_model_name, " from Hugging Face model ", huggingface_model_id)
    main()
