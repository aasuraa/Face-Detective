import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime

class Alert:
    def __init__(self):
        self.ACCESS_KEY = ''
        self.SECRET_KEY = '/'

    def sendText(self, pName):
        faceURL = "https://facedetective-2020.s3-us-west-2.amazonaws.com/lastFace.jpg"
        uploaded = self.upload_to_aws("./output/lastFace.jpg", "facedetective-2020", "lastFace.jpg")
        if uploaded:
            client = boto3.client(
                "sns",
                aws_access_key_id=self.ACCESS_KEY,
                aws_secret_access_key=self.SECRET_KEY,
                region_name="us-east-1"
            )
            client.publish(
                PhoneNumber="+17814924960",
                # PhoneNumber="+61432848561",
                Message="ALERT! " + pName + " seen.. " + faceURL
            )
            print("[MSG] Text sent...")
            return datetime.now().minute

    def upload_to_aws(self, local_file, bucket, s3_file):
        s3 = boto3.client('s3', aws_access_key_id=self.ACCESS_KEY,
                          aws_secret_access_key=self.SECRET_KEY)

        try:
            s3.upload_file(local_file, bucket, s3_file)
            print("Upload Successful")
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False
