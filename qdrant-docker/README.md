# Qdrant Vector Database Setup

This directory contains the Docker Compose configuration for running Qdrant locally. Qdrant is a vector similarity search engine that we use to store and query embeddings of Notion pages.

## Features

- Latest version of Qdrant
- Persistent storage in iCloud
- Health checks
- Automatic restart
- Exposed REST API and GRPC ports

## Prerequisites

- Docker
- Docker Compose
- iCloud Drive enabled

## Usage

1. Start Qdrant:
   ```bash
   docker-compose up -d
   ```

2. Check if it's running:
   ```bash
   docker-compose ps
   ```

3. View logs:
   ```bash
   docker-compose logs -f
   ```

4. Stop Qdrant:
   ```bash
   docker-compose down
   ```

## Configuration

- REST API is available at `http://localhost:6333`
- GRPC interface is available at `localhost:6334`
- Data is persisted in iCloud at `~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data/qdrant-storage`
- Health check runs every 30 seconds

## Ports

- 6333: REST API
- 6334: GRPC interface

## Storage

Data is persisted in your iCloud Drive at `~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data/qdrant-storage`. This provides several benefits:

- Automatic backup to iCloud
- Sync across your devices
- Recovery in case of local machine failure
- Easy access to data files through iCloud Drive

Note: Make sure you have enough iCloud storage space, as vector databases can grow large depending on usage.

### First-Time Setup

Before starting Qdrant, create the storage directory in iCloud:
```bash
mkdir -p "~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data/qdrant-storage"
```

## Health Checks

The container includes health checks that:
- Run every 30 seconds
- Timeout after 10 seconds
- Retry 3 times before marking as unhealthy

## Troubleshooting

1. If the container fails to start, check logs:
   ```bash
   docker-compose logs qdrant
   ```

2. If you need to reset the storage:
   ```bash
   # Stop the container
   docker-compose down
   
   # Remove the storage directory
   rm -rf "~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data/qdrant-storage"
   
   # Recreate the directory
   mkdir -p "~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data/qdrant-storage"
   
   # Start fresh
   docker-compose up -d
   ```

3. To check if Qdrant is responding:
   ```bash
   curl http://localhost:6333/healthz
   ```

### iCloud-Specific Issues

1. If you see sync issues:
   - Check iCloud Drive status in System Preferences
   - Ensure you have enough iCloud storage space
   - Wait for any pending syncs to complete before starting Qdrant

2. If the storage path is not found:
   - Verify iCloud Drive is enabled
   - Check if the path exists: `ls "~/Library/Mobile Documents/com~apple~CloudDocs/wandering-rag-data"`
   - Create the directory manually if needed 