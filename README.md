# FNMA Data Product Discovery App

## Overview

### Project Description
The FNMA Data Product Discovery App is an AI-powered search interface that helps business users and data analysts discover relevant data products across Fannie Mae's data catalog. Built on Databricks and powered by a RAG (Retrieval-Augmented Generation) agent, the app translates natural language queries into structured searches across mortgage, housing, property, servicing, fraud, investor, and credit-risk datasets. Users can search for data products, view ranked results with metadata, and request access—all through an intuitive Streamlit interface.

### Key Capabilities
* **Natural Language Search**: Query the data catalog using plain English (e.g., "Find data products about borrower profiles")
* **AI-Powered Ranking**: Results are ranked by semantic relevance, data completeness, freshness, business value, and ownership transparency
* **Interactive Data Discovery**: View detailed metadata including schemas, table names, descriptions, and completeness scores
* **Access Request Workflow**: Request access to data products directly from the UI with integrated approval tracking
* **Dual Interface**: Data discovery tab for catalog search and chatbot tab for conversational assistance

---

## Architecture and Databricks Concepts

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App (app.py)                   │
│  ┌──────────────────┐         ┌──────────────────────────┐ │
│  │ Data Discovery   │         │ Chatbot Interface        │ │
│  │ Tab              │         │ (Placeholder)            │ │
│  └──────────────────┘         └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           model_serving_utils.py (API Layer)                │
│  • query_endpoint() - Non-streaming queries                 │
│  • query_endpoint_stream() - Streaming support              │
│  • parse_rag_response() - JSON extraction                   │
│  • is_endpoint_supported() - Endpoint validation            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Databricks Model Serving Endpoint              │
│  • RAG Agent (ResponsesAgent with OpenAI client)            │
│  • Vector Search Tool (product catalog index)               │
│  • LLM Endpoint (databricks-gpt-5)                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Unity Catalog Data Layer                     │
│  • Vector Search Index:                                     │
│    fnma_product_catalog_jcg.default.                        │
│    product_catalog_vector_index                             │
│  • Metadata Tables: Product_Name, Description,              │
│    table_names, TAG_NAME, TAG_VALUE                         │
└─────────────────────────────────────────────────────────────┘
```

### Databricks-Specific Components

**Workspace Objects:**
* **Databricks App**: Deployed as a Lakehouse App with Streamlit runtime
* **Model Serving Endpoint**: RAG agent endpoint (referenced via `SERVING_ENDPOINT` env var)
* **Unity Catalog**: Data governance layer for vector search index and metadata tables

**Data Layer:**
* **Catalog**: `fnma_product_catalog_jcg` (or `bircatalog` for legacy data)
* **Schema**: `default` (or `fnma` for legacy)
* **Vector Search Index**: `product_catalog_vector_index` - Contains embeddings of data product metadata, schemas, and descriptions
* **Key Columns**:
  * `Product_Name`: Fully qualified name of the data product
  * `Description`: Business-friendly explanation of the data product
  * `table_names`: List of tables included in the data product
  * `TAG_NAME` / `TAG_VALUE`: Metadata tags for classification
  * `__db_Description_vector`: Vector embeddings for semantic search

**Compute:**
* App runs on Databricks App Compute (serverless)
* Model serving endpoint uses dedicated serving infrastructure
* No direct cluster dependencies for the app itself

---

## Local and Workspace Setup

### Prerequisites

**Accounts & Access:**
* Databricks workspace access with SSO enabled
* Unity Catalog permissions:
  * `USE CATALOG` on `fnma_product_catalog_jcg` (or `bircatalog`)
  * `USE SCHEMA` on `default` (or `fnma`)
  * `SELECT` on vector search index table
* Model Serving permissions:
  * `CAN QUERY` on the RAG agent serving endpoint
* Databricks Apps feature enabled (GA as of DBR 14.3+)

**Required Workspace Resources:**
* A deployed RAG agent serving endpoint (see "Data, Models, and Dependencies" section)
* Vector search index populated with data product metadata

**Tools:**
* Python 3.9+ (3.10 recommended)
* Git
* Databricks CLI (optional but recommended)
* IDE: VS Code with Databricks extension (recommended) or PyCharm

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd data-product-discovery-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure authentication:**
   
   **Option A: Databricks CLI (Recommended)**
   ```bash
   databricks configure --token
   # Enter your workspace URL and personal access token
   ```

   **Option B: Environment Variables**
   ```bash
   export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
   export DATABRICKS_TOKEN="dapi..."
   ```

   **Important**: Never commit tokens to git. Use environment variables or the Databricks CLI profile.

5. **Set the serving endpoint:**
   ```bash
   export SERVING_ENDPOINT="your-rag-endpoint-name"
   ```

### Workspace Deployment Setup

**Method 1: Databricks Repos (Recommended)**

1. In your Databricks workspace, navigate to **Repos**
2. Click **Add Repo** and connect to your Git repository
3. The app will sync automatically with your repo

**Method 2: Manual Upload**

1. Navigate to **Workspace** → **Users** → **your-email**
2. Create a folder: `data-product-discovery-app`
3. Upload `app.py`, `model_serving_utils.py`, `app.yaml`, and `requirements.txt`

**Configure App Settings:**

1. Create a **Databricks App** from the workspace UI:
   * Go to **Apps** → **Create App**
   * Select the folder containing your app files
   * Choose `app.yaml` as the configuration file

2. **Add Serving Endpoint Resource:**
   * In the app configuration, add a resource named `serving-endpoint`
   * Select your RAG agent serving endpoint
   * Grant `CAN_QUERY` permission

3. **Environment Variables** (configured in `app.yaml`):
   * `SERVING_ENDPOINT`: Auto-populated from the serving endpoint resource
   * `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Set to `false` to disable telemetry

4. **Secrets** (if needed for access requests):
   * The app uses `st.context.headers.get('X-Forwarded-Access-Token')` for authentication
   * No additional secrets configuration required for basic functionality

---

## Running and Using the App

### Running Locally

1. **Ensure environment is configured** (see Local Development Setup)

2. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Access the app:**
   * Open your browser to `http://localhost:8501`
   * The app will authenticate using your Databricks CLI profile or environment variables

4. **Local Development Notes:**
   * The app will query the actual serving endpoint in your workspace
   * Ensure your local machine can reach the Databricks workspace (VPN if required)
   * Changes to `app.py` will auto-reload in Streamlit

### Running in Databricks

1. **Navigate to Apps** in your Databricks workspace

2. **Find your deployed app** in the list

3. **Click "Open App"** to launch the Streamlit interface

4. **Compute Selection:**
   * The app runs on **Databricks App Compute** (serverless)
   * No manual cluster selection required
   * Compute auto-scales based on usage

### Basic Usage Tour

**Data Discovery Tab:**

1. **Search Interface:**
   * Enter a natural language query in the search box (e.g., "borrower credit data")
   * Click the 🔍 Search button

2. **Results Display:**
   * **Query Understanding**: Shows how the AI interpreted your search
   * **Ranked Results**: Up to 3 data products displayed in cards
     * Rank badges (🥇 🥈 🥉)
     * Data product name and full identifier
     * Description and completeness score
     * Health status indicator
     * Table names in an embedded DataFrame
   * **Request Access Button**: Click to submit an access request for the data product

3. **Recommended Action:**
   * AI-generated suggestion on how to use the top data products together

**Chatbot Tab:**
* Placeholder for future conversational AI features
* Currently echoes user input (implement custom logic in `get_chatbot_response()`)

---

## Configuration and Environments

### Configuration Model

**Primary Configuration File: `app.yaml`**
```yaml
command: ["streamlit", "run", "app.py"]
env:
  - name: STREAMLIT_BROWSER_GATHER_USAGE_STATS
    value: "false"
  - name: "SERVING_ENDPOINT"
    valueFrom: "serving-endpoint"  # References app resource
```

**Environment Variables:**
* `SERVING_ENDPOINT`: Name of the RAG agent serving endpoint (required)
* `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Disable Streamlit telemetry (optional)

**Runtime Configuration (in `app.py`):**
* `SERVING_ENDPOINT`: Retrieved from `os.getenv('SERVING_ENDPOINT')`
* Databricks SDK Config: Auto-configured via `databricks.sdk.core.Config()`

### Environment Strategy

**Development / Test / Production Separation:**

| Environment | Workspace | Catalog | Serving Endpoint | App Deployment |
|-------------|-----------|---------|------------------|----------------|
| **Dev** | `dev-workspace` | `fnma_product_catalog_jcg_dev` | `rag-agent-dev` | Local or Dev App |
| **Test** | `test-workspace` | `fnma_product_catalog_jcg_test` | `rag-agent-test` | Test App |
| **Prod** | `prod-workspace` | `fnma_product_catalog_jcg` | `rag-agent-prod` | Prod App |

**Promotion Process:**
1. Update `app.yaml` to reference the correct serving endpoint resource name
2. Update vector search index references in the serving endpoint configuration
3. Deploy the app to the target workspace
4. Test thoroughly before promoting to production

**Configuration Changes Between Environments:**
* Serving endpoint name (via app resource configuration)
* Vector search index name (configured in the serving endpoint, not the app)
* Unity Catalog names (if using separate catalogs per environment)

### Secrets and Credentials

**Required Secrets:**
* **Access Token for API Calls**: Used in `request_access_api()` function
  * Retrieved via `st.context.headers.get('X-Forwarded-Access-Token')`
  * Automatically provided by Databricks Apps runtime
  * No manual secret configuration needed

**Optional Secrets (for future enhancements):**
* External API keys (if integrating with non-Databricks services)
* Service principal credentials (for automated workflows)

**How to Create/Store Secrets:**
```bash
# Using Databricks CLI
databricks secrets create-scope --scope my-app-secrets
databricks secrets put --scope my-app-secrets --key api-token
```

**Accessing Secrets in Code:**
```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
secret_value = w.dbutils.secrets.get(scope="my-app-secrets", key="api-token")
```

---

## Data, Models, and Dependencies

### Data Contracts

**Input Data:**
* **Vector Search Index**: `fnma_product_catalog_jcg.default.product_catalog_vector_index`
  * **Schema**:
    * `unique_id` (STRING): Unique identifier for the data product
    * `Product_Name` (STRING): Fully qualified name (catalog.schema.table)
    * `Description` (STRING): Business-friendly description
    * `TAG_NAME` (STRING): Metadata tag name
    * `TAG_VALUE` (STRING): Metadata tag value
    * `table_names` (STRING): JSON array of table names
    * `__db_Description_vector` (ARRAY<FLOAT>): Vector embeddings
  * **SLA**: Index should be refreshed daily to reflect new data products
  * **Ownership**: Data Engineering team

**Output Data:**
* **Access Requests**: Submitted via Databricks API (`/api/2.0/rfa/request`)
  * Payload includes: `comment`, `securable` (catalog/schema), `privileges`
  * Response logged to app session state

### ML Models

**RAG Agent Model:**
* **Model Name**: Registered in Unity Catalog (e.g., `fnma_product_catalog_jcg.default.demo_data_dicovery_rag`)
* **Model Type**: MLflow `ResponsesAgent` with OpenAI client
* **Serving Endpoint**: Deployed as a model serving endpoint
* **LLM Backend**: `databricks-gpt-5` (or `databricks-claude-sonnet-4-5`)
* **Tools**:
  * `VectorSearchRetrieverTool`: Searches the product catalog vector index
  * Columns retrieved: `Product_Name`, `Description`, `table_names`, `TAG_NAME`, `TAG_VALUE`

**Model Versioning:**
* Models are versioned in Unity Catalog
* Serving endpoint can be updated to point to new model versions
* App does not need code changes when model versions change

**Model Dependencies:**
* Vector search index must exist and be accessible
* LLM endpoint must be available and have sufficient capacity
* Unity Catalog permissions must be granted to the serving endpoint

### External Services

**Databricks APIs:**
* **Model Serving API**: Used by `mlflow.deployments.get_deploy_client()`
  * Endpoint: `https://<workspace>/serving-endpoints/<endpoint-name>/invocations`
  * Authentication: Databricks token (auto-handled by SDK)
* **Request for Access API**: Used in `request_access_api()`
  * Endpoint: `https://<workspace>/api/2.0/rfa/request`
  * Authentication: Bearer token from app context

**No External (Non-Databricks) Services:**
* All functionality is self-contained within Databricks

---

## Development Workflow

### Branching and Git Workflow

**Branching Model:**
* `main`: Production-ready code
* `develop`: Integration branch for features
* `feature/<feature-name>`: Individual feature branches
* `hotfix/<issue>`: Emergency fixes for production

**Workflow:**
1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/new-search-filter
   ```

2. Make changes and commit:
   ```bash
   git add .
   git commit -m "Add filter for data product type"
   ```

3. Push and create a pull request:
   ```bash
   git push origin feature/new-search-filter
   # Create PR on GitHub/GitLab targeting 'develop'
   ```

4. After review and approval, merge to `develop`

5. Periodically merge `develop` to `main` for production releases

**Commit Conventions:**
* Use descriptive commit messages
* Prefix with type: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
* Example: `feat: add completeness score filter to search results`

### Databricks-Specific Dev Practices

**Repos vs Local Development:**
* **Use Repos for**:
  * Collaborative development
  * Automatic sync with Git
  * Testing in the actual Databricks environment
* **Use Local Dev for**:
  * Rapid iteration with hot-reload
  * Debugging with IDE tools
  * Offline development

**Testing Strategy:**

**Unit Tests (Pure Python Logic):**
```python
# test_model_serving_utils.py
import pytest
from model_serving_utils import parse_rag_response

def test_parse_rag_response_with_output_key():
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"text": '{"query_understanding": "test", "results": []}'}]
            }
        ]
    }
    result = parse_rag_response(response)
    assert result is not None
    assert "query_understanding" in result
```

**Integration Tests (In Databricks):**
* Create a test notebook that imports and calls app functions
* Use a test serving endpoint with known data
* Validate end-to-end flow from query to response parsing

**Manual Testing:**
* Deploy to a dev app instance
* Test with real queries and verify results
* Check access request workflow

### Coding Standards

**Style Guidelines:**
* Follow PEP 8 for Python code
* Use type hints where possible (e.g., `def query_endpoint(endpoint_name: str, messages: list[dict[str, str]]) -> dict:`)
* Maximum line length: 120 characters
* Use docstrings for all functions (Google style)

**Folder Structure:**
```
data-product-discovery-app/
├── app.py                      # Main Streamlit application
├── model_serving_utils.py      # API utilities for model serving
├── app.yaml                    # Databricks App configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── tests/                      # Unit and integration tests (future)
│   ├── test_app.py
│   └── test_model_serving_utils.py
└── .gitignore                  # Git ignore file
```

**Code Organization:**
* **`app.py`**: UI logic, Streamlit components, session state management
* **`model_serving_utils.py`**: Reusable functions for querying endpoints, parsing responses
* Keep business logic separate from UI code for testability

**Notebooks vs Python Modules:**
* Use Python modules (`.py` files) for production code
* Use notebooks for exploratory analysis, data validation, and documentation
* Link notebooks to Git repos for version control

---

## Troubleshooting

### Common Issues

**1. "Unable to determine serving endpoint" Error**
* **Cause**: `SERVING_ENDPOINT` environment variable not set
* **Fix**: 
  * Local: `export SERVING_ENDPOINT="your-endpoint-name"`
  * Databricks App: Ensure serving endpoint resource is configured in app settings

**2. "Vector search endpoint not found" Error**
* **Cause**: Incorrect vector search index name in serving endpoint configuration
* **Fix**: Update the index name in the RAG agent's `agent.py` file (Cell 5) to match the actual index

**3. "RESOURCE_ALREADY_EXISTS" Error**
* **Cause**: Trying to create an endpoint that already exists
* **Fix**: This is handled by the app's exception handling. If you see this, the existing endpoint will be used.

**4. Access Request Fails**
* **Cause**: Invalid token or insufficient permissions
* **Fix**: Ensure the app has access to the `X-Forwarded-Access-Token` header and the user has permissions to request access

**5. Results Not Displaying**
* **Cause**: RAG response parsing failed
* **Fix**: Check logs for parsing errors. The response format may have changed. Update `parse_rag_response()` in `model_serving_utils.py`

### Debugging Tips

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Inspect Raw Responses:**
* Add `st.write(response)` in `app.py` to display raw endpoint responses
* Check the `raw_response` key in parsed results

**Test Endpoint Directly:**
```python
from model_serving_utils import query_endpoint
response = query_endpoint("your-endpoint", [{"role": "user", "content": "test query"}])
print(response)
```

---

## Deployment and Operations

### Deployment Checklist

- [ ] Code reviewed and merged to `main` branch
- [ ] All tests passing
- [ ] Serving endpoint deployed and tested
- [ ] Vector search index populated and accessible
- [ ] App configuration updated for target environment
- [ ] Secrets configured (if needed)
- [ ] App deployed to Databricks workspace
- [ ] Smoke test completed (search query returns results)
- [ ] Access request workflow tested
- [ ] Monitoring and logging configured

### Monitoring

**App Health:**
* Monitor app uptime via Databricks Apps dashboard
* Check for errors in Streamlit logs (accessible via app UI)

**Serving Endpoint Health:**
* Monitor endpoint metrics in Databricks Model Serving UI
* Track query latency, error rates, and throughput

**Usage Metrics:**
* Track search queries and access requests via app logs
* Analyze popular search terms to improve data product metadata

### Maintenance

**Regular Tasks:**
* **Weekly**: Review app logs for errors
* **Monthly**: Update dependencies (`pip install --upgrade -r requirements.txt`)
* **Quarterly**: Review and update vector search index with new data products

**Scaling Considerations:**
* App compute auto-scales with Databricks Apps
* Serving endpoint can be scaled up/down based on query volume
* Vector search index size impacts query latency—consider partitioning for large catalogs

---

## Contributing

### How to Contribute

1. Fork the repository (if external) or create a feature branch
2. Make your changes following the coding standards
3. Add tests for new functionality
4. Update documentation (README, docstrings)
5. Submit a pull request with a clear description of changes

### Code Review Guidelines

* All PRs require at least one approval
* Reviewers should check:
  * Code quality and adherence to standards
  * Test coverage
  * Documentation updates
  * Security considerations (no hardcoded secrets)

---

## License and Support

**License:** [Specify license, e.g., MIT, Apache 2.0, or proprietary]

**Support:**
* For bugs and feature requests, open an issue in the repository
* For Databricks-specific questions, contact your Databricks account team
* Internal support: [Specify internal Slack channel or email]

---

## Appendix

### Useful Links

* [Databricks Apps Documentation](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html)
* [MLflow Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
* [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
* [Vector Search](https://docs.databricks.com/generative-ai/vector-search.html)
* [Streamlit Documentation](https://docs.streamlit.io/)

### Glossary

* **RAG**: Retrieval-Augmented Generation - AI technique combining search with LLM generation
* **Vector Search**: Semantic search using vector embeddings
* **Unity Catalog**: Databricks' unified data governance solution
* **Model Serving**: Databricks service for deploying ML models as REST APIs
* **Lakehouse App**: Databricks Apps for deploying interactive applications

### Version History

* **v1.0.0** (2025-01-XX): Initial release with data discovery and access request features
* **v0.9.0** (2024-12-XX): Beta release for internal testing

---

**Last Updated:** 2025-01-16  
**Maintained By:** Joy Garnett (joy.garnett@databricks.com or garne041@gmail.com) and Anjana Sriram (anjana.sriram@databricks.com)
