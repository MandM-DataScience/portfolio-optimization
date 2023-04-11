# documentation https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
# https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

# ACCESS TO AWS S3
from configparser import ConfigParser

import boto3
from botocore.exceptions import ClientError
import logging
import json
import os

from portfolio_optimization import PORTFOLIO_BASE_DIR

"""
aws ecr get-login-password | docker login --username AWS --password-stdin 429663024407.dkr.ecr.eu-south-1.amazonaws.com
"""
def get_session():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    aws_access_key_id = parser.get("s3", "aws_access_key_id")
    aws_secret_access_key = parser.get("s3", "aws_secret_access_key")
    aws_region = parser.get("s3", "aws_region")

    return boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )


def get_s3_client():
    session = get_session()
    s3_client = session.client('s3')
    return s3_client


def get_s3_bucket():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    BUCKET_ID = parser.get("s3", "BUCKET_ID")

    session = get_session()
    s3 = session.resource('s3')
    return s3.Bucket(BUCKET_ID)


def list_s3_bucket_objects(params):

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    BUCKET_ID = parser.get("s3", "BUCKET_ID")

    s3_client = get_s3_client()
    objects = s3_client.list_objects_v2(Bucket=BUCKET_ID, **params)
    if 'Contents' in objects:
        return [obj['Key'] for obj in objects['Contents']]
    return []


def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    if not os.path.exists(file_name):
        logging.error(f"File Path {file_name} does not exist")
        return
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = get_s3_client()
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        print(json.dumps(response, indent=2))
    except ClientError as e:
        logging.error(e)
        return False
    return True


def put_object_to_s3(data, bucket, object_name):
    """Put an object to an S3 bucket
        :param data: Data to upload, must be in byte, string or file-like format
        :param bucket: Bucket to upload to
        :param object_name: S3 object name
        :return: True if file was uploaded, else False
    """
    s3_client = get_s3_client()
    try:
        response = s3_client.put_object(Body=data, Bucket=bucket, Key=object_name)
        print(response)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def download_file_from_s3(object_name, bucket, file_name=None):
    """Get a file from an S3 bucket and save it

        :param object_name: Object to get
        :param bucket: Bucket to download from
        :param file_name: Path to save downloaded file
    """
    if file_name is None:
        file_name = f"download/{object_name}"
    s3_client = get_s3_client()
    try:
        response = s3_client.download_file(bucket, object_name, file_name)
    except ClientError as e:
        logging.error(e)


def get_object_from_s3(object_name, bucket):
    """Get an object from an S3 bucket
        :param object_name: Object to get
        :param bucket: Bucket to download from
        :return: object downloaded or None if not found
    """
    s3_client = get_s3_client()
    result = None
    try:
        response = s3_client.get_object(Bucket=bucket, Key=object_name)
        if 'Body' in response:
            result = response['Body'].read()
    except ClientError as e:
        logging.error(e)
        return None
    return result


def delete_object_from_s3(object_name, bucket):
    """Delete a single object from an S3 bucket
        :param object_name: Object to delete
        :param bucket: Bucket to delete from
        :return: True if deleted else False
    """
    s3_client = get_s3_client()
    try:
        response = s3_client.delete_object(Bucket=bucket, Key=object_name)
        print(response)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def delete_multiple_object_from_s3(bucket, prefix=''):
    """Delete a single object from an S3 bucket
        :param object_name: Object to delete
        :param bucket: Bucket to delete from
        :param prefix: Prefix to filter objects in the bucket
        :return: True if deleted else False
    """
    s3_client = get_s3_client()
    try:
        object_keys = list_s3_bucket_objects({"Prefix": prefix})
        delete_keys = {'Objects': [{"Key": k} for k in object_keys]}
        print(delete_keys)
        response = s3_client.delete_objects(Bucket=bucket, Delete=delete_keys)
        print(response)
    except ClientError as e:
        logging.error(e)
        return False
    return True