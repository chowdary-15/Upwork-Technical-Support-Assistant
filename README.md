# Upwork Technical Support Assistant — Enterprise RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) chatbot assistant explicitly engineered to process technical documentation datasets and resolve complex queries regarding Upwork's modern API specifications, OAuth2 security models, and GraphQL schemas.

## 🚀 Key Architectural Features
- **Deterministic Tokenization & Chunking**: Natively implements recursive page-by-page extractions via `pypdf`, indexing documentation layouts into granular, 500-character chunks with a 50-character sliding array overlap.
- **Hardware-Isolated Vector Store**: Leverages an integrated `FAISS` vector database mapped directly onto localized system CPU instructions. This avoids runtime execution crashes caused by virtual PyTorch meta-tensor initialization flaws.
- **Enterprise Hallucination Guardrail**: Integrates a strict zero-shot prompt verification layer. If an inquiry is out-of-scope or unverified by the extracted chunks, the assistant returns a clean, mandated security fallback phrase.
- **Observable Optimization**: Features transparent logging matrix outputs and latency tracking gauges processing production-level inferences in ~1.0 seconds.

## 🛠️ Quick Start Instructions

### 1. Configure the Local Environment
Ensure you have Python 3.10 installed on your system. Install all project dependency requirements directly using pip:
```bash
pip install -r requirements.txt