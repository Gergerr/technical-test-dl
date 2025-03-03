import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account

PROJECT_ID = "datalabs-test-452308"
LOCATION = "us-central1"
# Replace with the *actual* path to your service account key file
CREDENTIALS_FILE = '/Users/geraldo/Documents/Code/MLOps/datalabs-techtest/askdata/service-account.json'

# Load credentials from the service account key file.
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)

# Initialize Vertex AI using the explicit credentials.
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

def create_embedding(text: str, model_name: str = "textembedding-gecko@003") -> list[float]:
    """Creates a text embedding."""
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    text = "This is a test sentence."
    try:
        embedding = create_embedding(text)
        print(f"Embedding length: {len(embedding)}")
    except Exception as e:
        print(f"Error: {e}")