# Vector Database Management Guide

This guide covers installation, running, and querying vector databases used in this project: **ChromaDB**, **Pinecone**, and **Milvus**, along with universal management tools: **Langflow**, **Flowise**, and **VectorAdmin**.

---

## Table of Contents

1. [ChromaDB (Local)](#chromadb-local)
2. [Pinecone (Cloud)](#pinecone-cloud)
3. [Milvus (Podman/Docker)](#milvus-podmandocker)
4. [Langflow - Universal Vector DB Management (Recommended)](#langflow---universal-vector-db-management-recommended)
5. [Flowise - Alternative Low-Code Platform](#flowise---alternative-low-code-platform)
6. [VectorAdmin - Legacy Universal Management Tool](#vectoradmin---legacy-universal-management-tool)
7. [Vector Database Management Tools Comparison](#vector-database-management-tools-comparison)
8. [Current Project Configuration](#current-project-configuration)

---

## ChromaDB (Local)

### What is ChromaDB?
ChromaDB is an open-source vector database designed for development speed and local prototyping. It's great for small to medium datasets (under 10M vectors) but not optimized for production workloads at 50M+ vectors.

### Installation
ChromaDB is already included in this project via LangChain:

```bash
pip install langchain-chroma
# or with uv
uv pip install langchain-chroma
```

### Running ChromaDB
ChromaDB runs embedded in your Python application - no separate server needed for the persistent storage mode used in this project.

### Current Configuration (from .env)
```bash
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=mediumblog_rag
```

### Using ChromaDB in this Project

**For ingestion** (run `ingestion.py`):
```bash
python ingestion.py
# Select option 2 - Chroma (local disk)
```

**For retrieval** (run `main.py`):
```bash
python main.py
# Select option 2 - Chroma (local)
```

### Querying ChromaDB Data

**Option 1: Python Script**
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="mediumblog_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Query
results = vectorstore.similarity_search("your query here", k=5)
for doc in results:
    print(doc.page_content)
    print("---")
```

**Option 2: ChromaDB Client**
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("mediumblog_rag")

# Get collection info
print(f"Collection count: {collection.count()}")

# Query
results = collection.query(
    query_texts=["your query"],
    n_results=5
)
print(results)
```

---

## Pinecone (Cloud)

### What is Pinecone?
Pinecone is a fully managed cloud vector database optimized for production workloads, offering high performance and scalability for billions of vectors.

### Installation
Pinecone is already included in this project:

```bash
pip install langchain-pinecone pinecone-client
# or with uv
uv pip install langchain-pinecone pinecone-client
```

### Setup & Configuration

1. **Create a Pinecone account** at [https://www.pinecone.io/](https://www.pinecone.io/)

2. **Get your API key** from the Pinecone console

3. **Create an index** (or use existing):
   - Dimension: 1536 (for OpenAI text-embedding-ada-002)
   - Metric: cosine
   - Cloud/Region: Choose based on your location

4. **Add to .env file**:
```bash
PINECONE_API_KEY=your_api_key_here
INDEX_NAME=your_index_name
```

### Current Configuration (from .env)
```bash
PINECONE_API_KEY=pcsk_7RWeWV_...
INDEX_NAME=medium-blogs-embeddings-index
```

### Using Pinecone in this Project

**For ingestion** (run `ingestion.py`):
```bash
python ingestion.py
# Select option 1 - Pinecone (cloud index)
```

**For retrieval** (run `main.py`):
```bash
python main.py
# Select option 1 - Pinecone
```

### Querying Pinecone Data

**Option 1: LangChain (used in this project)**
```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name="medium-blogs-embeddings-index",
    embedding=embeddings
)

results = vectorstore.similarity_search("your query", k=5)
for doc in results:
    print(doc.page_content)
```

**Option 2: Pinecone Client directly**
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your_api_key")
index = pc.Index("medium-blogs-embeddings-index")

# Get index stats
print(index.describe_index_stats())

# Query with vector
query_vector = [0.1, 0.2, ...]  # 1536 dimensions
results = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True
)
print(results)
```

---

## Milvus (Podman/Docker)

### What is Milvus?
Milvus is an open-source vector database designed for similarity search at scale. It can handle billions of vectors and offers high performance with advanced features like hybrid search and GPU acceleration.

### Installation & Setup

#### Method 1: Quick Start with Installation Script (Recommended)

1. **Download the installation script**:
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```

2. **Start Milvus**:
```bash
bash standalone_embed.sh start
```

This creates a Podman/Docker container named `milvus` running on:
- **Milvus port**: 19530
- **Embedded etcd**: 2379

3. **Stop Milvus**:
```bash
bash standalone_embed.sh stop
```

#### Method 2: Podman Compose (For Production-like Setup)

**Note**: Using Podman instead of Docker. Commands are nearly identical: `podman-compose` replaces `docker-compose`, and `podman` replaces `docker`.

1. **Install podman-compose** (if not already installed):
```bash
# On macOS
brew install podman-compose

# On Linux (Fedora/RHEL)
sudo dnf install podman-compose

# On Linux (Ubuntu/Debian)
pip install podman-compose
```

2. **Create `docker-compose.yml`** (works with both Docker and Podman):
```yaml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.4.0
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

3. **Start services with Podman**:
```bash
# Start in detached mode
podman-compose up -d

# Or view logs while starting
podman-compose up
```

4. **Verify services are running**:
```bash
# Check running containers
podman ps

# Check specific service logs
podman-compose logs milvus-standalone
```

5. **Stop services**:
```bash
# Stop and remove containers
podman-compose down

# Stop without removing
podman-compose stop
```

6. **Useful Podman commands**:
```bash
# List all containers (running and stopped)
podman ps -a

# View resource usage
podman stats

# Inspect a container
podman inspect milvus-standalone

# Remove all stopped containers
podman container prune
```

### Python Integration with Milvus

1. **Install Milvus Python SDK**:
```bash
pip install pymilvus
# or with uv
uv pip install pymilvus
```

2. **Add to .env**:
```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

3. **Example Usage**:
```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain_openai import OpenAIEmbeddings

# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Create a collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields=fields, description="Document embeddings")
collection = Collection(name="mediumblog_rag", schema=schema)

# Create index
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

# Insert data
embeddings_model = OpenAIEmbeddings()
texts = ["your text here"]
vectors = embeddings_model.embed_documents(texts)

collection.insert([vectors, texts])
collection.load()

# Query
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
query_vector = embeddings_model.embed_query("your query")

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=5,
    output_fields=["text"]
)

for hits in results:
    for hit in hits:
        print(f"Score: {hit.score}, Text: {hit.entity.get('text')}")
```

### Using LangChain with Milvus

```bash
pip install langchain-milvus
```

```python
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Initialize Milvus vector store
vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name="mediumblog_rag",
    connection_args={
        "host": "localhost",
        "port": "19530"
    }
)

# Add documents
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("./mediumblog1.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore.add_documents(docs)

# Query
results = vectorstore.similarity_search("your query", k=5)
for doc in results:
    print(doc.page_content)
```

---

## Langflow - Universal Vector DB Management (Recommended)

### What is Langflow?
Langflow is an **open-source, low-code visual platform** for building AI applications and RAG workflows. It provides a **drag-and-drop interface** to manage multiple vector databases (ChromaDB, Pinecone, Milvus, Qdrant, pgvector, and more) without writing code.

**Status**: ✅ **Actively maintained** (47,000+ GitHub stars, last updated 2026)

### Why Choose Langflow?
- **Vendor-agnostic**: Works with ChromaDB, Pinecone, Milvus, Qdrant, pgvector, AstraDB
- **Visual workflow builder**: Drag-and-drop components for vector stores
- **Built on LangChain**: Same framework used in this project
- **Open Source**: Apache 2.0 License
- **Production-ready**: Used for building production AI applications
- **Full CRUD operations**: Read, write, query, and manage vector data
- **No code required**: Visual interface for all operations

### Features
- Visual vector database component configuration
- Query vector databases and inspect results as JSON or tables
- Connect multiple vector databases simultaneously
- Build RAG pipelines visually
- Integrate with LLMs (OpenAI, Claude, Ollama, etc.)
- Export workflows as Python code
- REST API for programmatic access
- Real-time flow debugging

---

### Installation Methods

#### Method 1: pip Install (Simplest)

1. **Install Langflow**:
```bash
# Create a virtual environment (recommended)
python -m venv langflow-env
source langflow-env/bin/activate  # On Windows: langflow-env\Scripts\activate

# Install Langflow
pip install langflow

# Or with uv
uv pip install langflow
```

2. **Run Langflow**:
```bash
langflow run
```

3. **Access the UI**:
   - Open browser to `http://localhost:7860`
   - Default port: 7860

4. **Custom port** (optional):
```bash
langflow run --port 8080
```

---

#### Method 2: Podman/Docker (Isolated Environment)

1. **Pull and run Langflow image**:
```bash
# Using Podman
podman run -d \
  --name langflow \
  -p 7860:7860 \
  -v langflow-data:/app/langflow \
  langflowai/langflow:latest

# Or using Docker
docker run -d \
  --name langflow \
  -p 7860:7860 \
  -v langflow-data:/app/langflow \
  langflowai/langflow:latest
```

2. **Access the UI**:
   - Open browser to `http://localhost:7860`

3. **Stop Langflow**:
```bash
podman stop langflow

# Or with Docker
docker stop langflow
```

4. **Restart Langflow**:
```bash
podman start langflow

# Or with Docker
docker start langflow
```

5. **View logs**:
```bash
podman logs -f langflow

# Or with Docker
docker logs -f langflow
```

---

#### Method 3: From Source (For Development)

1. **Clone repository**:
```bash
git clone https://github.com/langflow-ai/langflow.git
cd langflow
```

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Run Langflow**:
```bash
langflow run
```

---

### Connecting Vector Databases to Langflow

#### General Workflow
1. Open Langflow UI at `http://localhost:7860`
2. Create a new flow or use a template
3. Drag a **Vector Store** component from the left sidebar
4. Configure the component with your database credentials
5. Connect to other components (embeddings, LLM, chat interface)

---

#### Connecting ChromaDB (Local)

1. **In Langflow UI**:
   - Drag the **"Chroma"** component from Vector Stores section
   - Click on the component to open configuration panel

2. **Configure ChromaDB component**:
   - **Collection Name**: `mediumblog_rag`
   - **Persist Directory**: `/Users/rajranja/Documents/github/rajiv-ranjan/langchain-course/chroma_db` (use absolute path)
   - **Embedding Function**: Connect to an **OpenAI Embeddings** component

3. **Add OpenAI Embeddings component**:
   - Drag **"OpenAI Embeddings"** from Embeddings section
   - **API Key**: Your OpenAI API key (from .env)
   - Connect the output to ChromaDB's embedding input

4. **Query the vector store**:
   - Add a **"Vector Store Retriever"** component
   - Connect ChromaDB output to the retriever
   - Set **k**: `5` (number of results)

**Example Flow**:
```
[OpenAI Embeddings] → [Chroma DB] → [Vector Store Retriever] → [Chat Output]
```

---

#### Connecting Pinecone (Cloud)

1. **In Langflow UI**:
   - Drag the **"Pinecone"** component from Vector Stores section
   - Click to configure

2. **Configure Pinecone component**:
   - **API Key**: `pcsk_7RWeWV_...` (from your .env)
   - **Index Name**: `medium-blogs-embeddings-index`
   - **Namespace**: (leave empty or specify if using namespaces)
   - **Embedding Function**: Connect to **OpenAI Embeddings**

3. **Add OpenAI Embeddings**:
   - Same as ChromaDB setup above

4. **Query and search**:
   - Use **Vector Store Retriever** component
   - Configure search parameters (top_k, filter, etc.)

**Example Flow**:
```
[OpenAI Embeddings] → [Pinecone] → [Vector Store Retriever] → [LLM Chain] → [Chat]
```

---

#### Connecting Milvus (Self-hosted)

**Prerequisites**: Milvus must be running on `localhost:19530` (see Milvus section above)

1. **In Langflow UI**:
   - Drag the **"Milvus"** component from Vector Stores section
   - Click to configure

2. **Configure Milvus component**:
   - **Connection URI**: `http://localhost:19530`
   - **Collection Name**: `mediumblog_rag`
   - **Embedding Function**: Connect to **OpenAI Embeddings**
   - **Index Parameters** (optional):
     ```json
     {
       "metric_type": "L2",
       "index_type": "IVF_FLAT",
       "params": {"nlist": 128}
     }
     ```

3. **Add OpenAI Embeddings**:
   - Drag **"OpenAI Embeddings"** component
   - Enter API key

4. **Create collection** (if it doesn't exist):
   - Langflow will auto-create the collection on first use
   - Or pre-create using Python (see Milvus section)

**Example Flow**:
```
[Document Loader] → [Text Splitter] → [OpenAI Embeddings] → [Milvus] → [Success]
```

---

### Common Langflow Workflows

#### 1. Ingest Documents into Vector Database

**Components needed**:
- File Loader / Text Loader
- Text Splitter (Character or Recursive)
- OpenAI Embeddings
- Vector Store (Chroma/Pinecone/Milvus)

**Flow**:
```
[File Upload] → [Text Splitter] → [OpenAI Embeddings] → [Chroma/Pinecone/Milvus]
```

**Steps**:
1. Drag **"File"** component (upload your document)
2. Connect to **"Character Text Splitter"**
   - Chunk Size: `1000`
   - Chunk Overlap: `200`
3. Connect to **"OpenAI Embeddings"**
4. Connect to your chosen **Vector Store**
5. Click **"Run"** to ingest

---

#### 2. Query Vector Database (RAG)

**Components needed**:
- Chat Input
- OpenAI Embeddings
- Vector Store Retriever
- Prompt Template
- LLM (ChatOpenAI or ChatOllama)
- Chat Output

**Flow**:
```
[Chat Input] → [Embeddings] → [Vector Store Retriever] → [Prompt] → [LLM] → [Chat Output]
```

**Steps**:
1. **Chat Input**: User query entry point
2. **Embeddings**: Convert query to vector
3. **Retriever**: Search vector database (connect to Chroma/Pinecone/Milvus)
4. **Prompt Template**:
   ```
   Answer based on context:
   
   Context: {context}
   
   Question: {question}
   
   Answer:
   ```
5. **LLM**: ChatOpenAI or ChatOllama
6. **Chat Output**: Display response

---

#### 3. Compare Results from Multiple Vector Databases

**Flow**:
```
[Query] → [Embeddings] → ┬─ [ChromaDB Retriever] → [Results 1]
                          ├─ [Pinecone Retriever] → [Results 2]
                          └─ [Milvus Retriever]   → [Results 3]
```

This allows you to compare search quality across different vector databases.

---

### Querying Vector Databases in Langflow

#### Visual Query (No Code)

1. **Open your flow** with a retriever component
2. **Click the component** you want to query
3. **Enter query text** in the input field
4. **Click "Run"** or **"Play"** button
5. **View results** in the output panel:
   - JSON format with documents and scores
   - Table format for easier reading

#### Programmatic Query (Python API)

Langflow provides a REST API to run flows programmatically:

```python
import requests
import json

# Langflow API endpoint
LANGFLOW_URL = "http://localhost:7860/api/v1/run"
FLOW_ID = "your-flow-id"  # Get from Langflow UI

# Query payload
payload = {
    "input_value": "What is Pinecone in machine learning?",
    "input_type": "chat",
    "output_type": "chat"
}

# Run the flow
response = requests.post(
    f"{LANGFLOW_URL}/{FLOW_ID}",
    json=payload,
    headers={"Content-Type": "application/json"}
)

# Get results
results = response.json()
print(json.dumps(results, indent=2))
```

#### Export as Python Code

1. **In Langflow UI**, click **"Export"** button
2. Select **"Python Code"**
3. Download the generated script
4. Run locally:
```bash
python exported_flow.py
```

---

### Environment Variables for Langflow

Create a `.env` file in your Langflow directory or set system variables:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key

# Langflow configuration
LANGFLOW_HOST=0.0.0.0
LANGFLOW_PORT=7860
LANGFLOW_DATABASE_URL=sqlite:///langflow.db

# Optional: Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

Langflow will auto-detect these variables and use them in components.

---

### Advanced Features

#### 1. Multi-Database Migration

**Use Case**: Move data from ChromaDB to Pinecone

**Flow**:
```
[ChromaDB Source] → [Get All Documents] → [Pinecone Destination] → [Insert]
```

#### 2. Hybrid Search

Combine multiple retrieval strategies:
```
[Query] → ┬─ [Vector Search (Semantic)]
          └─ [Keyword Search (BM25)]
          
[Combine Results] → [Rerank] → [LLM] → [Answer]
```

#### 3. Vector Database Analytics

**Flow**:
```
[Vector Store] → [Get Stats] → [Visualize] → [Dashboard]
```

Get collection count, dimension info, index statistics.

---

### Managing Multiple Projects

Langflow supports multiple flows (projects):

1. **Create new flow**: Click "New Flow"
2. **Save flow**: Auto-saves, or click "Save"
3. **Switch flows**: Use the dropdown menu
4. **Export/Import flows**: Share flows as JSON files

**Organized by use case**:
- `chromadb_ingestion.json` - For ChromaDB data loading
- `pinecone_rag.json` - For Pinecone RAG queries
- `milvus_search.json` - For Milvus search testing

---

### Troubleshooting

#### Issue: "Cannot connect to vector database"
- **ChromaDB**: Verify absolute path to `chroma_db` directory
- **Pinecone**: Check API key and index name
- **Milvus**: Ensure Milvus container is running (`podman ps | grep milvus`)

#### Issue: "Embedding dimension mismatch"
- Ensure all components use the same embedding model
- OpenAI embeddings = 1536 dimensions
- Check vector database index dimension matches

#### Issue: "Module not found"
- Install missing dependencies: `pip install langchain-<provider>`
- Example: `pip install langchain-chroma langchain-pinecone langchain-milvus`

#### Issue: Langflow won't start
- Check port 7860 is not in use: `lsof -i :7860`
- Use custom port: `langflow run --port 8080`

---

### Documentation & Resources

- **Official Website**: [https://www.langflow.org/](https://www.langflow.org/)
- **GitHub Repository**: [https://github.com/langflow-ai/langflow](https://github.com/langflow-ai/langflow) (47k+ stars)
- **Documentation**: [https://docs.langflow.org/](https://docs.langflow.org/)
- **Vector Stores Guide**: [https://docs.langflow.org/components-vector-stores/](https://docs.langflow.org/components-vector-stores/)
- **Community**: Discord, GitHub Discussions

### License
Apache 2.0 - Fully open source

---

## Flowise - Alternative Low-Code Platform

### What is Flowise?
Flowise is an **open-source, low-code platform** built with TypeScript for creating AI agents and workflows. Like Langflow, it provides a **visual drag-and-drop interface** for managing vector databases, but with a focus on AI agent orchestration.

**Status**: ✅ **Actively maintained** (53,254+ GitHub stars, last updated May 2026)

### Why Choose Flowise?
- **Vendor-agnostic**: Supports Pinecone, ChromaDB, Qdrant, Weaviate, and 100+ integrations
- **TypeScript-based**: Different architecture from Langflow (Python)
- **AI Agent focus**: Specialized for building conversational agents
- **Open Source**: Free tier available, enterprise features for advanced use cases
- **Human-in-the-loop**: HITL execution for agent workflows
- **Self-hosted or cloud**: Flexible deployment options

### Features
- Node-based visual workflow builder
- RAG pipeline creation with vector stores
- Multi-agent orchestration
- Vector database CRUD operations
- Role-based access control (RBAC)
- Execution tracing and debugging
- API endpoints for each flow
- Integration with 100+ LLMs and tools

---

### Installation Methods

#### Method 1: npm Install (Simplest)

1. **Prerequisites**:
```bash
# Ensure Node.js is installed (v18 or higher)
node --version

# Ensure npm is installed
npm --version
```

2. **Install Flowise globally**:
```bash
npm install -g flowise
```

3. **Run Flowise**:
```bash
npx flowise start
```

4. **Access the UI**:
   - Open browser to `http://localhost:3000` (default port)

5. **Custom configuration**:
```bash
# Custom port
npx flowise start --PORT=8080

# Custom database path
npx flowise start --FLOWISE_DATABASE_PATH=/path/to/database
```

---

#### Method 2: Podman/Docker (Recommended for Isolation)

1. **Pull and run Flowise image**:
```bash
# Using Podman
podman run -d \
  --name flowise \
  -p 3000:3000 \
  -v flowise-data:/root/.flowise \
  flowiseai/flowise:latest

# Or using Docker
docker run -d \
  --name flowise \
  -p 3000:3000 \
  -v flowise-data:/root/.flowise \
  flowiseai/flowise:latest
```

2. **With environment variables**:
```bash
# Using Podman
podman run -d \
  --name flowise \
  -p 3000:3000 \
  -e FLOWISE_USERNAME=admin \
  -e FLOWISE_PASSWORD=your_password \
  -e OPENAI_API_KEY=your_openai_key \
  -v flowise-data:/root/.flowise \
  flowiseai/flowise:latest
```

3. **Access the UI**:
   - Open browser to `http://localhost:3000`

4. **Stop/Start/Restart**:
```bash
# Stop
podman stop flowise

# Start
podman start flowise

# Restart
podman restart flowise

# View logs
podman logs -f flowise
```

5. **Remove container**:
```bash
podman rm -f flowise

# Remove volume as well
podman volume rm flowise-data
```

---

#### Method 3: Podman Compose (Multi-Container Setup)

1. **Create `docker-compose.yml`**:
```yaml
version: '3.8'

services:
  flowise:
    image: flowiseai/flowise:latest
    container_name: flowise
    ports:
      - "3000:3000"
    volumes:
      - flowise-data:/root/.flowise
    environment:
      - FLOWISE_USERNAME=admin
      - FLOWISE_PASSWORD=flowise123
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    restart: unless-stopped

volumes:
  flowise-data:
```

2. **Start with Podman Compose**:
```bash
podman-compose up -d
```

3. **View logs**:
```bash
podman-compose logs -f flowise
```

4. **Stop**:
```bash
podman-compose down
```

---

### Connecting Vector Databases to Flowise

#### General Workflow
1. Open Flowise UI at `http://localhost:3000`
2. Click **"Add New Chatflow"** or **"Add New Agentflow"**
3. Drag **Vector Store** nodes from the left panel
4. Click node to configure credentials
5. Connect to Document Loaders, Embeddings, and LLMs
6. Save and test the flow

---

#### Connecting ChromaDB (Local)

1. **In Flowise UI**:
   - Click **"+ Add Node"** or drag from sidebar
   - Select **"Vector Stores"** → **"Chroma"**

2. **Configure Chroma node**:
   - **Chroma URL**: (leave empty for embedded mode)
   - **Collection Name**: `mediumblog_rag`
   - **Chroma Path**: `/Users/rajranja/Documents/github/rajiv-ranjan/langchain-course/chroma_db`
   - **Embeddings**: Connect to **OpenAI Embeddings** node

3. **Add OpenAI Embeddings node**:
   - Drag **"Embeddings"** → **"OpenAI Embeddings"**
   - **API Key**: Your OpenAI API key
   - Connect output to Chroma's embeddings input

4. **Add Document Loader** (for ingestion):
   - Drag **"Document Loaders"** → **"Text File"**
   - Upload or specify path to `mediumblog1.txt`
   - Connect to **Text Splitter** → **Chroma**

**Example Chatflow**:
```
[Text File] → [Text Splitter] → [OpenAI Embeddings] → [Chroma] → [Conversational Retrieval]
```

---

#### Connecting Pinecone (Cloud)

1. **In Flowise UI**:
   - Drag **"Vector Stores"** → **"Pinecone"**

2. **Configure Pinecone node**:
   - **Pinecone API Key**: `pcsk_7RWeWV_...` (from .env)
   - **Pinecone Index**: `medium-blogs-embeddings-index`
   - **Pinecone Namespace**: (optional, leave empty or specify)
   - **Embeddings**: Connect to **OpenAI Embeddings**

3. **Add OpenAI Embeddings**:
   - Same as ChromaDB setup

4. **Create RAG chatflow**:
   ```
   [Conversation] → [Pinecone Retriever] → [LLM Chain] → [Chat Response]
   ```

**Advanced Configuration**:
- **Top K**: Number of results (default: 4)
- **Metadata Filter**: JSON filter for specific documents
  ```json
  {
    "source": "mediumblog1.txt"
  }
  ```

---

#### Connecting Milvus (Self-hosted)

**Prerequisites**: Milvus running on `localhost:19530`

1. **In Flowise UI**:
   - Drag **"Vector Stores"** → **"Milvus"**

2. **Configure Milvus node**:
   - **Milvus Server URL**: `http://localhost:19530`
   - **Collection Name**: `mediumblog_rag`
   - **Embeddings**: Connect to **OpenAI Embeddings**
   - **Milvus Database**: `default` (or your database name)

3. **Index Configuration** (optional):
   - **Metric Type**: `L2` or `IP` (Inner Product) or `COSINE`
   - **Index Type**: `IVF_FLAT`, `IVF_SQ8`, or `HNSW`

4. **Create ingestion flow**:
   ```
   [PDF Loader] → [Recursive Text Splitter] → [OpenAI Embeddings] → [Milvus]
   ```

5. **Create query flow**:
   ```
   [Chat Input] → [Milvus Retriever] → [Prompt Template] → [ChatOpenAI] → [Output]
   ```

---

### Common Flowise Workflows

#### 1. Document Ingestion (Upsert to Vector DB)

**Nodes**:
- Document Loader (Text/PDF/CSV)
- Text Splitter
- Embeddings (OpenAI/Hugging Face)
- Vector Store (Chroma/Pinecone/Milvus)

**Flow**:
```
[Document Loader] → [Recursive Character Text Splitter] → [OpenAI Embeddings] → [Pinecone]
                                                              ↓
                                                        [Upsert Complete]
```

**Configuration**:
- **Text Splitter**: Chunk Size: 1000, Overlap: 200
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: Auto-upsert on flow execution

---

#### 2. Conversational RAG (Query with Memory)

**Nodes**:
- Chat Input
- Conversational Retrieval Chain
- Vector Store Retriever
- LLM (ChatOpenAI/ChatOllama)
- Memory (Buffer/Summary)
- Chat Output

**Flow**:
```
[Chat Input] → [Conversational Retrieval] ← [Pinecone Retriever]
                      ↓                           ↑
                [Memory Buffer]              [OpenAI Embeddings]
                      ↓
                [ChatOpenAI]
                      ↓
                [Chat Output]
```

**Features**:
- Maintains conversation history
- Retrieves relevant context from vector DB
- Combines chat memory with RAG

---

#### 3. Multi-Vector Store Comparison

**Flow**:
```
[User Query] → [OpenAI Embeddings] → ┬─ [ChromaDB Retriever] → [Results A]
                                     ├─ [Pinecone Retriever] → [Results B]
                                     └─ [Milvus Retriever]   → [Results C]
                                              ↓
                                        [Combine & Rank]
                                              ↓
                                            [LLM]
```

Compare search quality and performance across databases.

---

#### 4. Agent with Vector Store Tool

**Nodes**:
- Agent (OpenAI Functions/ReAct)
- Vector Store as Tool
- Calculator Tool
- Web Search Tool
- LLM

**Flow**:
```
[Agent] ← [Vector Store Tool (Pinecone)]
   ↓      [Calculator Tool]
   ↓      [Web Search Tool]
   ↓
[ChatOpenAI]
   ↓
[Response]
```

Agent decides when to query vector database vs. other tools.

---

### Querying Vector Databases in Flowise

#### Visual Query (UI-based)

1. **Open your chatflow**
2. **Click "Test Chatflow"** button (bottom-right)
3. **Enter query** in the chat input
4. **View results**:
   - Retrieved documents shown in debug panel
   - Similarity scores displayed
   - Final LLM response in chat

#### API Query (Programmatic)

Each Flowise chatflow exposes a REST API:

```python
import requests
import json

# Flowise API endpoint
FLOWISE_URL = "http://localhost:3000/api/v1/prediction"
CHATFLOW_ID = "your-chatflow-id"  # Get from Flowise UI URL

# Query payload
payload = {
    "question": "What is Pinecone in machine learning?",
    "overrideConfig": {
        "returnSourceDocuments": True
    }
}

# Execute query
response = requests.post(
    f"{FLOWISE_URL}/{CHATFLOW_ID}",
    json=payload,
    headers={"Content-Type": "application/json"}
)

# Parse results
results = response.json()
print("Answer:", results.get("text"))
print("\nSource Documents:")
for doc in results.get("sourceDocuments", []):
    print(f"- {doc['pageContent'][:100]}... (Score: {doc['metadata'].get('score')})")
```

#### Webhook Integration

Flowise can trigger webhooks on completion:

```bash
# Configure in Flowise UI
Webhook URL: https://your-server.com/webhook
Method: POST
Headers: {"Authorization": "Bearer token"}
```

---

### Environment Variables for Flowise

Create a `.env` file or set environment variables:

```bash
# Flowise Configuration
FLOWISE_USERNAME=admin
FLOWISE_PASSWORD=your_secure_password
PORT=3000
FLOWISE_DATABASE_PATH=/path/to/flowise.db

# Vector Database Credentials
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

# Optional: Milvus
MILVUS_URL=http://localhost:19530

# Optional: Authentication
FLOWISE_SECRETKEY_OVERWRITE=your_secret_key
```

Flowise automatically loads credentials from environment variables.

---

### Advanced Features

#### 1. Custom Tools
Create custom tools that query your vector databases:

```javascript
// Custom Pinecone search tool
{
  name: "pinecone_search",
  description: "Search technical documents in Pinecone",
  parameters: {
    query: "string"
  }
}
```

#### 2. Multi-Step Agents
Build agents that orchestrate multiple vector database queries:

```
[Agent] → Step 1: Search ChromaDB (local cache)
       → Step 2: If not found, search Pinecone (cloud)
       → Step 3: Update ChromaDB with new findings
```

#### 3. Analytics & Monitoring
- **Execution logs**: Track all queries and responses
- **Token usage**: Monitor API costs
- **Latency metrics**: Measure vector search performance

---

### Managing Multiple Chatflows

Flowise organizes work into chatflows:

1. **Create**: Click "Add New Chatflow"
2. **Name**: Give descriptive names
   - `chromadb_ingestion`
   - `pinecone_rag_chat`
   - `milvus_search_agent`
3. **Export/Import**: Share as JSON files
4. **Version control**: Track changes in git

**API Access**:
Each chatflow gets a unique endpoint:
```
http://localhost:3000/api/v1/prediction/<chatflow-id>
```

---

### Troubleshooting

#### Issue: "Cannot connect to Chroma"
- **Solution**: Verify absolute path to `chroma_db`
- Check file permissions (Flowise needs read/write access)

#### Issue: "Pinecone authentication failed"
- **Solution**: Verify API key is correct
- Check index exists: `pinecone.list_indexes()`

#### Issue: "Milvus connection refused"
- **Solution**: Ensure Milvus container running:
  ```bash
  podman ps | grep milvus
  ```
- Check Milvus URL: `http://localhost:19530` (not `https`)

#### Issue: Port 3000 already in use
- **Solution**: Change port:
  ```bash
  npx flowise start --PORT=8080
  ```
- Or stop other service using port 3000

#### Issue: "Module not installed"
- **Solution**: Flowise auto-installs dependencies, but if issues persist:
  ```bash
  npm install -g flowise
  ```

---

### Documentation & Resources

- **Official Website**: [https://flowiseai.com/](https://flowiseai.com/)
- **GitHub Repository**: [https://github.com/FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) (53k+ stars)
- **Documentation**: [https://docs.flowiseai.com/](https://docs.flowiseai.com/)
- **Vector Stores Guide**: [https://docs.flowiseai.com/integrations/langchain/vector-stores](https://docs.flowiseai.com/integrations/langchain/vector-stores)
- **Community**: Discord, GitHub Discussions

### License
Open Source (Apache 2.0), with enterprise features for advanced use cases

---

## VectorAdmin - Legacy Universal Management Tool

### What is VectorAdmin?
VectorAdmin is an open-source universal GUI and management tool for vector databases. It provides a unified interface to manage multiple vector databases (Pinecone, ChromaDB, Milvus, Qdrant, Weaviate) simultaneously.

**⚠️ Important Note**: As of April 2026, the VectorAdmin project has been **archived and is no longer actively maintained**. 

**Recommendation**: Use **Langflow** or **Flowise** instead (see sections above). VectorAdmin is included here for legacy reference only.

### Features
- Multi-user web interface
- Manage multiple vector databases from one dashboard
- Migrate data between different vector databases
- Semantic search through UI
- Document upload and embedding
- Developer API
- No changes required to existing pipelines

### Installation Methods

#### Method 1: Podman/Docker (Recommended - Simplest)

1. **Clone the repository**:
```bash
git clone https://github.com/Mintplex-Labs/vector-admin.git
cd vector-admin
```

2. **Using Podman Compose** (see `docker/DOCKER.md` in the repo):
```bash
cd docker

# With Podman
podman-compose up -d

# Or with Docker
docker-compose up -d
```

3. **Access the web interface**:
   - Open browser to `http://localhost:3000` (or configured port)
   - First-time setup will redirect you to create admin account

4. **Stop services**:
```bash
# With Podman
podman-compose down

# Or with Docker
docker-compose down
```

#### Method 2: Development Setup (Advanced)

**Prerequisites**:
- Node.js and yarn
- Python 3.9+
- OpenAI API key (for embeddings/document uploads)
- Running vector database instance (ChromaDB, Pinecone, or Milvus)

**Installation Steps**:

1. **Clone and setup**:
```bash
git clone https://github.com/Mintplex-Labs/vector-admin.git
cd vector-admin
yarn dev:setup
```

2. **Setup Python environment for document processor**:
```bash
cd document-processor
python3.9 -m venv v-env
source v-env/bin/activate  # On Windows: v-env\Scripts\activate
pip install -r requirements.txt
cd ..
```

3. **Setup database**:
```bash
yarn prisma:setup
```

4. **Start services** (in separate terminal windows from project root):

   **Terminal 1 - Backend Server**:
   ```bash
   yarn dev:server
   ```

   **Terminal 2 - Frontend**:
   ```bash
   yarn dev:frontend
   ```

   **Terminal 3 - Background Workers**:
   ```bash
   yarn dev:workers
   ```

   **Terminal 4 - Document Processor**:
   ```bash
   cd document-processor
   source v-env/bin/activate
   flask run --host '0.0.0.0' --port 8888
   ```

5. **Access the application**:
   - Open `http://localhost:3000`
   - Create admin account on first visit
   - Configure organization and database connections

### Configuration

#### Environment Variables

**Disable telemetry** (optional):
```bash
export DISABLE_TELEMETRY="true"
```

**For connecting to your databases**, configure through the web UI:
- Navigate to Settings → Database Connections
- Add connection details for each vector database

### Connecting Vector Databases to VectorAdmin

#### For ChromaDB (Local):
1. In VectorAdmin UI, go to "Add Connection"
2. Select "ChromaDB"
3. Configure:
   - **Type**: Persistent Client
   - **Path**: `./chroma_db` (absolute path to this project's ChromaDB directory)
   - **Collection Name**: `mediumblog_rag`

#### For Pinecone (Cloud):
1. In VectorAdmin UI, go to "Add Connection"
2. Select "Pinecone"
3. Configure:
   - **API Key**: Your Pinecone API key (from .env)
   - **Environment**: Your Pinecone environment
   - **Index Name**: `medium-blogs-embeddings-index`

#### For Milvus (Docker):
1. Ensure Milvus is running on `localhost:19530`
2. In VectorAdmin UI, go to "Add Connection"
3. Select "Milvus"
4. Configure:
   - **Host**: `localhost`
   - **Port**: `19530`
   - **Collection Name**: `mediumblog_rag`

### Using VectorAdmin

Once connected, you can:
- **Browse collections**: View all vectors and metadata
- **Search**: Perform semantic search across your data
- **Upload documents**: Add new documents through the UI
- **Migrate data**: Move data between different vector databases
- **Monitor**: Track usage and performance
- **API access**: Use the Developer API for programmatic access

### Documentation & Resources

- **Official Documentation**: [https://mintplex-labs-inc.gitbook.io/vectoradmin-by-mintplex-labs/](https://mintplex-labs-inc.gitbook.io/vectoradmin-by-mintplex-labs/)
- **GitHub Repository**: [https://github.com/Mintplex-Labs/vector-admin](https://github.com/Mintplex-Labs/vector-admin)
- **Website**: [https://vectoradmin.com/](https://vectoradmin.com/)

### System Requirements
- **Windows**: Not supported for development environment
- **Docker**: Supported on all platforms
- **Multi-user**: Yes, with permission management
- **Cloud deployment**: Ready for cloud hosting

### License
MIT License - Free to use and modify

---

## Vector Database Management Tools Comparison

### Quick Decision Guide

**Choose Langflow if you want:**
- ✅ Most actively maintained (47k+ stars)
- ✅ Built on LangChain (same as this project)
- ✅ Python-based, familiar ecosystem
- ✅ Best for RAG workflows
- ✅ Visual flow builder for data pipelines
- ✅ Easiest to export to Python code

**Choose Flowise if you want:**
- ✅ TypeScript-based architecture
- ✅ Focus on AI agent orchestration
- ✅ Human-in-the-loop workflows
- ✅ More enterprise features (RBAC, audit logs)
- ✅ Larger integration ecosystem (100+ tools)
- ✅ Better for complex multi-agent systems

**Choose VectorAdmin if you:**
- ⚠️ Have legacy installations only
- ⚠️ Project archived (April 2026)
- ❌ Not recommended for new projects

---

### Feature Comparison Table

| Feature | Langflow | Flowise | VectorAdmin (Legacy) |
|---------|----------|---------|----------------------|
| **Status** | ✅ Active (2026) | ✅ Active (2026) | ⚠️ Archived (Apr 2026) |
| **GitHub Stars** | 47,000+ | 53,254+ | ~10,000 (archived) |
| **License** | Apache 2.0 | Apache 2.0 | MIT |
| **Language** | Python | TypeScript | Node.js + Python |
| **Framework** | LangChain | LangChain | Custom |
| **UI Type** | Visual flow builder | Node-based builder | Traditional GUI |
| **ChromaDB** | ✅ Full support | ✅ Full support | ✅ Full support |
| **Pinecone** | ✅ Full support | ✅ Full support | ✅ Full support |
| **Milvus** | ✅ Full support | ✅ Full support | ✅ Full support |
| **Qdrant** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Weaviate** | ✅ Yes | ✅ Yes | ✅ Yes |
| **pgvector** | ✅ Yes | ✅ Yes | ❌ No |
| **Installation** | `pip install` | `npm install` | Git clone + setup |
| **Container** | Podman/Docker | Podman/Docker | Docker only |
| **Default Port** | 7860 | 3000 | 3000 |
| **RAG Workflows** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Great | ⭐⭐⭐ Good |
| **AI Agents** | ⭐⭐⭐⭐ Great | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐ Limited |
| **Document Upload** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Vector Search** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Data Migration** | ✅ Yes | ✅ Yes | ✅ Yes |
| **REST API** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Export to Code** | ✅ Python | ❌ No | ❌ No |
| **Multi-user** | ✅ Yes | ✅ Yes (RBAC) | ✅ Yes |
| **Authentication** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cloud Hosting** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Learning Curve** | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Easy |
| **Documentation** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Community** | Very active | Very active | Inactive |
| **Best For** | RAG pipelines, Python devs | AI agents, TypeScript devs | Legacy only |

---

### Installation Comparison

| Method | Langflow | Flowise | VectorAdmin |
|--------|----------|---------|-------------|
| **pip/npm** | `pip install langflow` | `npm install -g flowise` | ❌ Not available |
| **Podman** | ✅ `podman run langflowai/langflow` | ✅ `podman run flowiseai/flowise` | ⚠️ Limited support |
| **Docker** | ✅ Yes | ✅ Yes | ✅ Yes |
| **From Source** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Setup Time** | ~2 minutes | ~2 minutes | ~10-15 minutes |

---

### Resource Usage Comparison

| Resource | Langflow | Flowise | VectorAdmin |
|----------|----------|---------|-------------|
| **Memory** | ~500MB | ~400MB | ~600MB |
| **CPU** | Low-Medium | Low-Medium | Medium |
| **Disk** | ~200MB | ~300MB | ~500MB |
| **Startup Time** | ~10 seconds | ~8 seconds | ~20 seconds |

---

### Use Case Recommendations

#### For This Project (ChromaDB + Pinecone + Milvus RAG):
**Primary Recommendation**: **Langflow**
- Built on LangChain (same framework)
- Easiest integration with existing code
- Great for RAG workflows
- Can export flows to Python
- Visual debugging of retrieval pipelines

**Alternative**: **Flowise**
- If you prefer TypeScript
- More enterprise features
- Better for complex agent workflows

**Avoid**: **VectorAdmin**
- Archived, no updates
- Use for legacy projects only

---

## Current Project Configuration

### Environment Variables (.env)
```bash
# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=medium-blogs-embeddings-index

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=mediumblog_rag

# Optional: Milvus (if using)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# LangSmith (for tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=project4-rag-gist
```

### Project Files
- **`ingestion.py`**: Ingests documents into Pinecone or ChromaDB
- **`main.py`**: RAG query application with both Pinecone and ChromaDB support
- **`mediumblog1.txt`**: Source document for embeddings

### Quick Start Workflow

1. **Choose your vector database**:
   - **Pinecone**: Cloud, production-ready, managed
   - **ChromaDB**: Local, fast development, free
   - **Milvus**: Self-hosted, scalable, advanced features

2. **Ingest documents**:
   ```bash
   python ingestion.py
   # Select your chosen database (1=Pinecone, 2=ChromaDB)
   ```

3. **Query with RAG**:
   ```bash
   python main.py
   # Select same database as ingestion
   # Select LLM (OpenAI or Ollama models)
   ```

4. **Optional: Use Langflow for visual management** (Recommended):
   ```bash
   # Install Langflow
   pip install langflow
   
   # Run Langflow
   langflow run
   
   # Access at http://localhost:7860
   # Create flows to query ChromaDB, Pinecone, or Milvus visually
   ```

5. **Alternative: Use Flowise for AI agent workflows**:
   ```bash
   # Install Flowise
   npm install -g flowise
   
   # Run Flowise
   npx flowise start
   
   # Access at http://localhost:3000
   ```

6. **Podman option for Langflow**:
   ```bash
   # Run Langflow in container
   podman run -d --name langflow -p 7860:7860 langflowai/langflow
   
   # Or Flowise
   podman run -d --name flowise -p 3000:3000 flowiseai/flowise
   ```

---

## Vector Database Technologies Comparison (2026)

| Feature | ChromaDB | Pinecone | Milvus |
|---------|----------|----------|--------|
| **Type** | Open-source, local | Managed cloud | Open-source, self-hosted |
| **Best For** | Development, small datasets | Production, managed | Production, large scale |
| **Scalability** | Up to ~10M vectors | Billions of vectors | Billions of vectors |
| **Cost** | Free | Pay per use | Infrastructure costs |
| **Setup Complexity** | Very easy | Easy | Moderate |
| **Performance** | Good for dev | Excellent | Excellent |
| **Hosting** | Local/embedded | Cloud-only | Self-hosted/cloud |
| **GPU Support** | No | Yes (managed) | Yes |
| **Production Ready** | No (dev only) | Yes | Yes |

---

## Additional Resources

### ChromaDB
- Documentation: [https://docs.trychroma.com/](https://docs.trychroma.com/)
- GitHub: [https://github.com/chroma-core/chroma](https://github.com/chroma-core/chroma)

### Pinecone
- Documentation: [https://docs.pinecone.io/](https://docs.pinecone.io/)
- Console: [https://app.pinecone.io/](https://app.pinecone.io/)

### Milvus
- Documentation: [https://milvus.io/docs](https://milvus.io/docs)
- Docker Hub: [https://hub.docker.com/r/milvusdb/milvus](https://hub.docker.com/r/milvusdb/milvus)
- Tutorials: [Milvus Tutorial: Vector DB RAG in 13 Steps](https://tech-insider.org/milvus-vector-database-tutorial-rag-13-steps-2026/)

### Langflow
- Official Website: [https://www.langflow.org/](https://www.langflow.org/)
- Documentation: [https://docs.langflow.org/](https://docs.langflow.org/)
- GitHub: [https://github.com/langflow-ai/langflow](https://github.com/langflow-ai/langflow) (47k+ stars)
- Vector Stores Guide: [https://docs.langflow.org/components-vector-stores/](https://docs.langflow.org/components-vector-stores/)

### Flowise
- Official Website: [https://flowiseai.com/](https://flowiseai.com/)
- Documentation: [https://docs.flowiseai.com/](https://docs.flowiseai.com/)
- GitHub: [https://github.com/FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) (53k+ stars)
- Vector Stores: [https://docs.flowiseai.com/integrations/langchain/vector-stores](https://docs.flowiseai.com/integrations/langchain/vector-stores)

### VectorAdmin (Legacy)
- Documentation: [https://mintplex-labs-inc.gitbook.io/vectoradmin-by-mintplex-labs/](https://mintplex-labs-inc.gitbook.io/vectoradmin-by-mintplex-labs/)
- GitHub: [https://github.com/Mintplex-Labs/vector-admin](https://github.com/Mintplex-Labs/vector-admin) (Archived)

### Podman Resources
- Podman Compose Guide: [https://www.datacamp.com/tutorial/podman-compose](https://www.datacamp.com/tutorial/podman-compose)
- Podman vs Docker 2026: [https://last9.io/blog/podman-vs-docker/](https://last9.io/blog/podman-vs-docker/)
- How to Choose Between Podman Compose and Docker Compose: [https://oneuptime.com/blog/post/2026-03-18-choose-between-podman-compose-docker-compose/view](https://oneuptime.com/blog/post/2026-03-18-choose-between-podman-compose-docker-compose/view)

### General Vector Database Resources
- [Best Vector Databases in 2026 - Encore](https://encore.dev/articles/best-vector-databases)
- [Best Vector Databases 2026 - DataCamp](https://www.datacamp.com/blog/the-top-5-vector-databases)
- [Vector Database Comparison 2026 - Reintech](https://reintech.io/blog/vector-database-comparison-2026-pinecone-weaviate-milvus-qdrant-chroma)
- [Flowise vs Langflow Comparison](https://www.aitooldiscovery.com/how-to/flowise-vs-langflow)

---

## Troubleshooting

### Langflow
- **Issue**: Port 7860 already in use
  - **Solution**: Use custom port: `langflow run --port 8080`
  - Find process: `lsof -i :7860`

- **Issue**: "Cannot connect to ChromaDB"
  - **Solution**: Use absolute path to chroma_db directory
  - Check permissions: Langflow needs read/write access

- **Issue**: "Module not found" for vector stores
  - **Solution**: Install dependencies:
    ```bash
    pip install langchain-chroma langchain-pinecone langchain-milvus
    ```

- **Issue**: Podman container won't start
  - **Solution**: Check logs: `podman logs langflow`
  - Ensure volume mounts are correct

### Flowise
- **Issue**: Port 3000 already in use
  - **Solution**: Change port: `npx flowise start --PORT=8080`
  - Find process: `lsof -i :3000`

- **Issue**: "Pinecone authentication failed"
  - **Solution**: Set environment variable:
    ```bash
    export PINECONE_API_KEY=your_key
    npx flowise start
    ```

- **Issue**: "Cannot connect to Milvus"
  - **Solution**: Verify Milvus URL is `http://localhost:19530` (not https)
  - Check Milvus is running: `podman ps | grep milvus`

- **Issue**: Dependencies not installing
  - **Solution**: Reinstall Flowise:
    ```bash
    npm uninstall -g flowise
    npm install -g flowise
    ```

### Podman/Docker
- **Issue**: "Permission denied" on volumes
  - **Solution**: Fix volume permissions:
    ```bash
    podman unshare chown -R 1000:1000 ./volumes
    ```

- **Issue**: podman-compose not found
  - **Solution**: Install podman-compose:
    ```bash
    # macOS
    brew install podman-compose
    
    # Linux
    pip install podman-compose
    ```

- **Issue**: Network issues between containers
  - **Solution**: Recreate network:
    ```bash
    podman-compose down
    podman network prune
    podman-compose up -d
    ```

### ChromaDB
- **Issue**: "Collection not found"
  - **Solution**: Run `ingestion.py` first to create the collection

- **Issue**: Empty results
  - **Solution**: Check `CHROMA_PERSIST_DIRECTORY` path is correct

### Pinecone
- **Issue**: "Index not found"
  - **Solution**: Create index in Pinecone console or verify `INDEX_NAME`

- **Issue**: Dimension mismatch
  - **Solution**: Ensure index dimension (1536) matches OpenAI embeddings

### Milvus
- **Issue**: "Connection refused"
  - **Solution**: Check container is running: `podman ps | grep milvus`
  - Restart if needed: `podman-compose restart milvus-standalone`

- **Issue**: Port already in use
  - **Solution**: Find process using port: `lsof -i :19530`
  - Stop other services or change Milvus port in compose file

### VectorAdmin
- **Issue**: Cannot connect to local ChromaDB
  - **Solution**: Use absolute path to `chroma_db` directory

- **Issue**: Services won't start
  - **Solution**: Check all prerequisites installed, ports not in use

---

## Summary

This guide covered:
- ✅ **3 Vector Databases**: ChromaDB (local), Pinecone (cloud), Milvus (self-hosted)
- ✅ **3 Management Tools**: Langflow (recommended), Flowise (alternative), VectorAdmin (legacy)
- ✅ **Podman Integration**: All Docker commands converted to Podman equivalents
- ✅ **Complete Workflows**: Install, run, query, and manage vector databases
- ✅ **Visual Management**: No-code interfaces for all databases

### Quick Start Recommendations

1. **For local development**: Use **ChromaDB** + **Langflow**
2. **For production**: Use **Pinecone** + **Langflow**
3. **For scale & self-hosted**: Use **Milvus** + **Langflow** or **Flowise**
4. **For AI agents**: Use **Flowise** with any vector database

### Next Steps

1. Install Langflow: `pip install langflow && langflow run`
2. Open browser to `http://localhost:7860`
3. Create a flow connecting to your existing ChromaDB or Pinecone
4. Build RAG pipelines visually
5. Export to Python code for production

---

**Last Updated**: June 16, 2026
