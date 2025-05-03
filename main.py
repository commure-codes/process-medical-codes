# main.py
import os
import sys # Keep for potential future debugging

# --- Debugging: Print Python executable and path ---
# print("--- Python Environment Debug ---")
# print(f"Executable: {sys.executable}")
# print("sys.path:")
# for p in sys.path:
#     print(f"  - {p}")
# print("--- End Debug ---")
# --- End Debugging ---

import io
import secrets
from typing import Dict, List, Optional

import pandas as pd
# Pinecone import
from pinecone import Pinecone
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
# Corrected Vertex AI imports using 'vertexai' namespace
from google.cloud import aiplatform
# Use the 'vertexai' namespace for language models
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
# Import specific exception for better handling
from google.api_core.exceptions import NotFound as GoogleNotFound
# from google.protobuf import struct_pb2 # No longer needed
from dotenv import load_dotenv
import time # For potential retries or delays

# --- Configuration & Initialization ---

# Load environment variables from .env file
load_dotenv()

# --- Environment Variable Checks ---
required_env_vars = [
    "BACKEND_USERNAME", "BACKEND_PASSWORD", "PINECONE_API_KEY",
    "PINECONE_ICD_INDEX_NAME", "PINECONE_CPT_INDEX_NAME",
    "GOOGLE_PROJECT_ID", "GOOGLE_LOCATION", "VERTEX_MODEL_NAME"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# --- FastAPI App ---
app = FastAPI(title="Medical Code Embedding Service")
security = HTTPBasic()

# --- Basic Authentication ---
def verify_credentials(credentials: HTTPBasicCredentials = Security(security)):
    """Verifies basic authentication credentials."""
    correct_username = secrets.compare_digest(credentials.username, os.getenv("BACKEND_USERNAME"))
    correct_password = secrets.compare_digest(credentials.password, os.getenv("BACKEND_PASSWORD"))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- Pinecone Initialization ---
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    print("Pinecone client initialized.")
    PINECONE_ICD_INDEX = os.getenv("PINECONE_ICD_INDEX_NAME")
    PINECONE_CPT_INDEX = os.getenv("PINECONE_CPT_INDEX_NAME")
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"Failed to initialize Pinecone client: {e}")


# --- Vertex AI SDK Initialization (Project/Location only) ---
# Model object will be loaded within get_embeddings
try:
    aiplatform.init(project=os.getenv("GOOGLE_PROJECT_ID"), location=os.getenv("GOOGLE_LOCATION"))
    print(f"Vertex AI SDK initialized for project {os.getenv('GOOGLE_PROJECT_ID')} in {os.getenv('GOOGLE_LOCATION')}")
except Exception as e:
    print(f"Error initializing Vertex AI SDK: {e}")
    raise RuntimeError(f"Failed to initialize Vertex AI SDK: {e}")

# --- Global variable for the embedding model (lazy loaded) ---
embedding_model: Optional[TextEmbeddingModel] = None
embedding_model_name: Optional[str] = None

# --- Embedding Function (Use TextEmbeddingModel via vertexai namespace) ---
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts using TextEmbeddingModel."""
    global embedding_model, embedding_model_name

    current_model_name = os.getenv("VERTEX_MODEL_NAME")

    # --- Lazy load the Model object if not loaded or name changed ---
    if embedding_model is None or embedding_model_name != current_model_name:
        try:
            print(f"Attempting to load Vertex AI TextEmbeddingModel: {current_model_name}")
            # creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            # if creds_path and not os.path.exists(creds_path):
            #     print(f"Warning: Service account key file not found at path: {creds_path}")

            # Use the specific class from the vertexai namespace
            embedding_model = TextEmbeddingModel.from_pretrained(current_model_name)
            embedding_model_name = current_model_name
            print(f"Vertex AI TextEmbeddingModel loaded successfully: {embedding_model_name}")

        except GoogleNotFound as e:
            print(f"Error: Vertex AI Model '{current_model_name}' not found (404). Check name, project, location, permissions, and API enablement.")
            embedding_model = None
            raise HTTPException(status_code=500, detail=f"Vertex AI Model '{current_model_name}' not found. Check configuration and permissions.") from e
        except Exception as e:
            print(f"Error loading Vertex AI TextEmbeddingModel: {e}")
            import traceback
            traceback.print_exc()
            embedding_model = None
            raise HTTPException(status_code=500, detail=f"Failed to load Vertex AI TextEmbeddingModel: {e}")

    # --- Proceed with embedding generation ---
    all_embeddings = []
    # Use the task type appropriate for storing embeddings for later retrieval
    task_type = "RETRIEVAL_DOCUMENT"
    batch_size = 250 # Adjust as needed

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            # Prepare inputs using TextEmbeddingInput, specifying the task type
            inputs = [TextEmbeddingInput(text=text, task_type=task_type) for text in batch_texts]
            embeddings_response = embedding_model.get_embeddings(inputs)

            batch_embeddings = [embedding.values for embedding in embeddings_response]
            all_embeddings.extend(batch_embeddings)
            print(f"Processed embedding batch starting at index {i}, got {len(batch_embeddings)} embeddings.")

        except Exception as e:
            print(f"Error getting embeddings from Vertex AI model for batch starting at index {i}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Vertex AI embedding prediction failed: {e}")

    print(f"Generated {len(all_embeddings)} embeddings in total.")
    return all_embeddings


# --- API Endpoint (No changes needed here) ---
@app.post("/upload/")
async def upload_and_embed(
    code_type: str = Form(...), # Expect 'ICD' or 'CPT'
    file: UploadFile = File(...),
    username: str = Depends(verify_credentials) # Protect endpoint
):
    """
    Receives an Excel file, processes descriptions, generates embeddings,
    and upserts them to the appropriate Pinecone index using Pinecone client v3+.
    """
    print(f"Received upload request for {code_type} from user {username}")

    # --- Determine Target Pinecone Index ---
    if code_type == "ICD":
        target_index_name = PINECONE_ICD_INDEX
    elif code_type == "CPT":
        target_index_name = PINECONE_CPT_INDEX
    else:
        raise HTTPException(status_code=400, detail="Invalid code_type. Must be 'ICD' or 'CPT'.")

    print(f"Target Pinecone index determined: {target_index_name}")

    # --- Check if target index exists in Pinecone ---
    existing_indexes = None
    try:
        print("Attempting to list Pinecone indexes...")
        index_list_obj = pc.list_indexes()

        if hasattr(index_list_obj, 'names') and callable(index_list_obj.names):
            existing_indexes = index_list_obj.names()
            print(f"Successfully listed indexes: {existing_indexes}")
        elif isinstance(index_list_obj, list):
             existing_indexes = index_list_obj
             print(f"Retrieved indexes as a direct list: {existing_indexes}")
        else:
            if isinstance(index_list_obj, dict) and 'indexes' in index_list_obj and isinstance(index_list_obj['indexes'], list):
                 existing_indexes = [idx.get('name') for idx in index_list_obj['indexes'] if idx.get('name')]
                 print(f"Extracted names from dict structure: {existing_indexes}")
            else:
                print(f"Warning: Unexpected type returned by pc.list_indexes(): {type(index_list_obj)}")
                raise TypeError(f"Cannot extract index names from type: {type(index_list_obj)}")

    except Exception as e:
         print(f"Error listing Pinecone indexes: {e}")
         import traceback
         traceback.print_exc()
         raise HTTPException(status_code=500, detail=f"Could not list Pinecone indexes: {e}")

    if existing_indexes is None:
         raise HTTPException(status_code=500, detail="Failed to retrieve list of Pinecone indexes.")

    if target_index_name not in existing_indexes:
        raise HTTPException(
            status_code=500,
            detail=f"Pinecone index '{target_index_name}' not found. Available indexes: {existing_indexes}. Please ensure it is created."
        )

    # --- Get the specific Pinecone Index object from the client ---
    try:
        index = pc.Index(target_index_name)
        print(f"Successfully connected to Pinecone index '{target_index_name}'.")
    except Exception as e:
        print(f"Error connecting to Pinecone index '{target_index_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Could not connect to Pinecone index: {e}")


    # --- File Validation and Reading ---
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel file (.xlsx or .xls).")

    try:
        contents = await file.read()
        excel_data = io.BytesIO(contents)
        df = pd.read_excel(excel_data, usecols=[0, 1], header=0, engine='openpyxl')
        df.columns = ['code', 'description']

        # --- Data Validation and Cleaning ---
        df.dropna(subset=['code', 'description'], inplace=True)
        df['code'] = df['code'].astype(str)
        df['description'] = df['description'].astype(str)
        df = df[df['code'].str.strip() != '']
        df = df[df['description'].str.strip() != '']

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid data found in the first two columns of the uploaded file.")

        print(f"Successfully read {len(df)} rows from {file.filename}")

        # --- Prepare data for embedding (Description only) ---
        texts_to_embed = [row['description'] for index, row in df.iterrows()]
        print(f"Prepared {len(texts_to_embed)} descriptions for embedding.")

        metadata = [
            {"code_type": code_type, "code": row['code'], "description": row['description']}
            for index, row in df.iterrows()
        ]
        vector_ids = [f"{code_type}-{row['code']}" for index, row in df.iterrows()]

        # --- Generate Embeddings ---
        print("Generating embeddings using Vertex AI (based on descriptions only)...")
        # This function call now uses the correct import and TextEmbeddingInput
        embeddings = get_embeddings(texts_to_embed)

        if len(embeddings) != len(df):
            raise HTTPException(status_code=500, detail=f"Mismatch in number of embeddings ({len(embeddings)}) and data rows ({len(df)}).")

        # --- Prepare for Pinecone Upsert ---
        vectors_to_upsert = list(zip(vector_ids, embeddings, metadata))

        # --- Upsert to the selected Pinecone index in batches ---
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index '{target_index_name}'...")
        batch_size = 100
        upserted_count = 0
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                upsert_response = index.upsert(vectors=batch)
                current_batch_upserted = len(batch)
                upserted_count += current_batch_upserted
                print(f"Upserted batch {i//batch_size + 1} ({current_batch_upserted} vectors) to '{target_index_name}'.")
            except Exception as e:
                print(f"Error upserting batch starting at index {i} to Pinecone index '{target_index_name}': {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Pinecone upsert failed for index '{target_index_name}': {e}")

        print(f"Successfully upserted {upserted_count} vectors to Pinecone index '{target_index_name}'.")
        return {"message": f"Successfully processed and embedded {upserted_count} {code_type} code descriptions into index '{target_index_name}'.", "filename": file.filename}

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded Excel file is empty or has no data in the first two columns.")
    except ValueError as ve:
         raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if file and not file.file.closed:
             await file.close()
             print("File closed.")


# --- Root Endpoint (Optional) ---
@app.get("/")
def read_root():
    return {"message": "Medical Code Embedding Service is running."}

# --- To run the backend (using uvicorn) ---
# Command: uvicorn main:app --reload --port 8000
