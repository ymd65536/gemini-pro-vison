from google.cloud import storage


def list_blobs(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket)
    return blobs
