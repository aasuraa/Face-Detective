import boto3

class TextAlert:
    def __int__(self):
        """
            create a client
        """
        self.client = boto3.client(
            "sns",
            aws_access_key_id="AKIASWSXDFK6L2BZF74C",
            aws_secret_access_key="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",
            region_name="us-east-1"
        )

    def notify(self):
        """
            Send sms
        """
        self.client.publish(
            PhoneNumber="+17814924960",
            Message="Hello World!"
        )