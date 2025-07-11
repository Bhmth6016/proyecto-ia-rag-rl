import os
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# 1. PATH VERIFICATION - Handle spaces in path
def get_credential_path():
    base_path = os.path.join(os.environ['USERPROFILE'], 'OneDrive', 'Documents', 'GitHub')
    credential_path = os.path.join(base_path, 'Nueva carpeta', 'proyecto-ia-rag-rl', 
                                'proyecto-ia-rag-rhlf-465606-0990bef3c314.json')
    
    # Verify path exists
    if not os.path.exists(credential_path):
        raise FileNotFoundError(f"Credential file not found at: {credential_path}")
    return credential_path

# 2. CREDENTIAL VALIDATION
def validate_credentials(credential_path):
    try:
        with open(credential_path) as f:
            creds = json.load(f)
        
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        if not all(field in creds for field in required_fields):
            raise ValueError("Service account JSON is missing required fields")
        
        return creds['project_id']  # Return verified project ID
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in service account file")

# 3. AUTHENTICATION TEST
def test_authentication(credential_path):
    credentials = service_account.Credentials.from_service_account_file(
        credential_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    
    # Refresh token to verify it works
    credentials.refresh(Request())
    return credentials

# MAIN EXECUTION
try:
    # Step 1: Get and verify path
    cred_path = get_credential_path()
    print(f"✓ Credentials path verified: {cred_path}")
    
    # Step 2: Validate JSON content
    project_id = validate_credentials(cred_path)
    print(f"✓ Service account JSON validated for project: {project_id}")
    
    # Step 3: Test authentication
    credentials = test_authentication(cred_path)
    print("✓ Google Cloud authentication successful")
    
    # Step 4: Initialize Vertex AI with all verified parameters
    vertexai.init(
        project=project_id,
        location="us-central1",
        credentials=credentials
    )
    
    # Step 5: Test Gemini
    model = GenerativeModel("gemini-pro")
    response = model.generate_content("What is the capital of France?")
    print("\n🎉 Success! Gemini response:")
    print(response.text)
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("\nTROUBLESHOOTING CHECKLIST:")
    print("1. Ensure the service account has 'Vertex AI User' role")
    print("2. Verify 'Vertex AI API' is enabled at: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com")
    print("3. Check your project billing is active")
    print("4. Try moving credentials to a simpler path (e.g., C:\\temp\\creds.json)")
    print("5. Test with gcloud auth: run 'gcloud auth application-default login'")