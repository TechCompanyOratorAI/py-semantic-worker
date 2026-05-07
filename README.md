# OratorAI Semantic Worker

Semantic analysis worker that compares transcript content with slides and topic information to provide presentation quality insights.

## Features

- **Content Relevance Analysis**: Compares transcript segments with presentation topic
- **Semantic Similarity**: Analyzes alignment between speech and slide content
- **Timing Alignment**: Checks if speech timing matches slide progression
- **Multi-language Support**: Supports Vietnamese and English text processing
- **Real-time Processing**: Polls SQS queue for analysis jobs
- **Comprehensive Scoring**: Provides detailed scores and suggestions

## Architecture

```
SQS Analysis Queue → Semantic Worker → Database + Webhook
                         ↓
                   Sentence Transformers
                   (Multilingual Embeddings)
```

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database access
- AWS SQS queue access
- Node API service for webhooks

### Setup

1. **Clone and navigate to directory**

```bash
cd py-semantic-worker
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run the worker**

```bash
python main.py
```

## Configuration

### Required Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-southeast-1
AWS_SQS_SEMANTIC_QUEUE_URL=https://sqs.region.amazonaws.com/account/queue-name

# Database Configuration
DATABASE_URL=postgresql://user:pass@host:port/db
# OR individual parameters:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=oratorai
DB_USER=username
DB_PASSWORD=password

# Webhook Configuration
WEBHOOK_BASE_URL=http://localhost:3000
WEBHOOK_SECRET=your_webhook_secret
```

### Optional Configuration

```bash
# Worker Settings
WORKER_ID=semantic-worker-1
POLL_INTERVAL=5
MAX_MESSAGES=1
WAIT_TIME_SECONDS=20

# Analysis Settings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENABLE_GPU=false
# EMBEDDING_DEVICE=auto
SIMILARITY_THRESHOLD=0.7
BATCH_SIZE=32

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=colored
```

## How It Works

### 1. Message Processing

The worker polls the SQS analysis queue for jobs containing:

```json
{
  "jobId": 123,
  "presentationId": 456,
  "metadata": {
    "triggeredBy": "asr_and_slides_completed"
  }
}
```

### 2. Data Retrieval

For each job, the worker fetches:

- Presentation details and topic information
- Transcript segments with timestamps
- Slide content (OCR extracted text)

### 3. Semantic Analysis

**Content Relevance Analysis**:

- Compares each transcript segment with topic name/description
- Uses multilingual sentence embeddings
- Identifies topic keywords mentioned
- Flags off-topic content

**Semantic Similarity Analysis**:

- Compares transcript segments with slide content
- Finds best matching slides for each segment
- Calculates cosine similarity scores
- Identifies content misalignment

**Timing Alignment Analysis**:

- Analyzes if speech timing matches slide progression
- Calculates expected slide for each segment
- Measures timing deviations
- Suggests pacing improvements

### 4. Results

The worker generates:

- **Segment-level scores**: Relevance, similarity, alignment for each segment
- **Overall scores**: Aggregated presentation quality metrics
- **Issues and suggestions**: Specific improvement recommendations
- **Detailed insights**: Keywords found, best matching slides, timing analysis

### 5. Webhook Delivery

Results are sent to the Node API via webhook:

```json
{
  "jobId": 123,
  "presentationId": 456,
  "status": "success",
  "analysis": {
    "segmentAnalyses": [...],
    "overallScores": {
      "contentRelevance": 0.78,
      "semanticSimilarity": 0.75,
      "slideAlignment": 0.72,
      "overallScore": 0.75
    },
    "metadata": {...}
  }
}
```

## Analysis Metrics

### Content Relevance (0.0 - 1.0)

- Measures how well transcript content relates to the presentation topic
- Uses semantic similarity between segment text and topic description
- Identifies topic keywords mentioned in speech

### Semantic Similarity (0.0 - 1.0)

- Measures alignment between speech and slide content
- Finds best matching slides for each transcript segment
- Detects when speaker content doesn't match displayed slides

### Slide Alignment (0.0 - 1.0)

- Measures timing alignment between speech and slide progression
- Calculates expected slide based on segment timing
- Identifies pacing issues (too fast/slow slide transitions)

### Overall Score (0.0 - 1.0)

- Weighted average of all metrics
- Provides single quality indicator
- Used for presentation ranking and comparison

## Docker Deployment

```bash
# Build image
docker build -t oratorai-semantic-worker .

# Run container
docker run -d \
  --name semantic-worker \
  --env-file .env \
  oratorai-semantic-worker
```

### GPU Embeddings

Set `ENABLE_GPU=true` to require GPU for sentence-transformer inference. With `ENABLE_GPU=false`, embeddings run on CPU unless `EMBEDDING_DEVICE` is set.

```bash
ENABLE_GPU=true
```

For advanced control, set `EMBEDDING_DEVICE`. If this is set, it overrides `ENABLE_GPU`:

```bash
# auto: use CUDA when available, otherwise CPU
EMBEDDING_DEVICE=auto

# cpu: force CPU
EMBEDDING_DEVICE=cpu

# cuda/gpu: require NVIDIA CUDA, fail fast if unavailable
EMBEDDING_DEVICE=cuda
```

For Docker Compose with GPU enabled:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The host must have an NVIDIA driver and NVIDIA Container Toolkit installed. The Python image also needs a CUDA-enabled `torch` build for `EMBEDDING_DEVICE=cuda`.

## Monitoring

### Health Checks

- Database connectivity
- SQS queue access
- Webhook endpoint reachability
- Model loading status

### Metrics

- Jobs processed/succeeded/failed
- Processing times
- Queue depth
- Error rates

### Logs

- Structured logging with timestamps
- Color-coded log levels
- Detailed error traces
- Performance metrics

## Troubleshooting

### Common Issues

**Model Loading Errors**:

```bash
# Check if model can be downloaded
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

**Database Connection Issues**:

```bash
# Test database connectivity
python -c "import psycopg2; psycopg2.connect('your_database_url')"
```

**SQS Permission Issues**:

- Ensure IAM user has `sqs:ReceiveMessage`, `sqs:DeleteMessage` permissions
- Check queue URL and region configuration

**Webhook Failures**:

- Verify webhook endpoint is accessible
- Check webhook secret configuration
- Monitor Node API logs for webhook processing errors

### Performance Tuning

**Memory Usage**:

- Adjust `BATCH_SIZE` based on available memory
- Use smaller embedding models for resource-constrained environments

**Processing Speed**:

- Increase `MAX_MESSAGES` for higher throughput
- Deploy multiple worker instances
- Use `ENABLE_GPU=true` or `EMBEDDING_DEVICE=auto/cuda` on GPU-enabled instances for faster embedding generation

## Development

### Project Structure

```
py-semantic-worker/
├── src/
│   ├── config/          # Configuration management
│   ├── services/        # Core services
│   │   ├── sqs_service.py
│   │   ├── database_service.py
│   │   ├── webhook_service.py
│   │   └── semantic_service.py
│   └── utils/           # Utilities
├── tests/               # Test files
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── Dockerfile         # Container definition
```

### Adding New Analysis Features

1. Extend `SemanticAnalysisService` class
2. Add new metrics to `SegmentAnalysis` dataclass
3. Update webhook payload format
4. Modify Node API webhook handler

### Testing

```bash
# Run tests
python -m pytest tests/

# Test individual components
python -c "from src.services.semantic_service import get_semantic_service; service = get_semantic_service()"
```

## License

Part of the OratorAI project - AI-powered presentation analysis platform.
