# RAG Assistant — Railway deployment

A Streamlit PDF question-answering app using Google Gemini embeddings and Gemini Flash.

## Run locally
1. Create a virtual environment and install dependencies: `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in your environment (or a local `.env` file with python-dotenv installed).
3. Run: `streamlit run app.py`

## Deploy to Railway
1. Push this folder to a new GitHub repository.
2. In Railway, create a project and deploy from that GitHub repository.
3. In the service's **Variables** tab, add `GEMINI_API_KEY` with your own Google AI Studio API key.
4. Railway should detect the Procfile. If needed, use the start command:
   `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`
5. In **Settings → Networking**, generate a public domain.

The PDF is uploaded and indexed in the running app session; it is not stored permanently.
Scanned/image-only PDFs need OCR and are not supported by this version.
