# 🐳 AEGIS Docker Deployment Guide

## Quick Start

### Windows
```bash
# Test Docker setup
docker-test.bat

# Start dashboard
run.bat

# Stop dashboard
stop.bat
```

### Linux/Mac
```bash
# Start dashboard
make docker-deploy

# Stop dashboard
make docker-down
```

Dashboard URL: **http://localhost:8501**

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Docker Container: cosmic-pipeline-app  │
│  ┌───────────────────────────────────┐  │
│  │  Streamlit Dashboard (Port 8501)  │  │
│  │  ├─ AEGIS UI (Turkish)            │  │
│  │  ├─ 3 Tabs (Data/Pipeline/Results)│  │
│  │  └─ Real-time Pipeline Execution  │  │
│  └───────────────────────────────────┘  │
│                                          │
│  Mounted Volumes:                        │
│  ├─ ./models → /app/models              │
│  ├─ ./data/cache → /app/data/cache      │
│  ├─ ./data/raw → /app/data/raw          │
│  └─ ./config → /app/config              │
└─────────────────────────────────────────┘
```

---

## Prerequisites

### 1. Install Docker Desktop
- **Windows**: https://www.docker.com/products/docker-desktop
- **Mac**: https://www.docker.com/products/docker-desktop
- **Linux**: https://docs.docker.com/engine/install/

### 2. Verify Installation
```bash
docker --version
docker compose version
```

### 3. Start Docker Daemon
- Windows/Mac: Open Docker Desktop
- Linux: `sudo systemctl start docker`

---

## Commands Reference

### Windows Batch Files

#### `run.bat`
- Creates required directories
- Checks for LSTM model (trains if missing)
- Builds Docker image
- Starts container
- Opens browser to http://localhost:8501

#### `stop.bat`
- Stops container
- Optional: Clean up Docker resources

#### `docker-test.bat`
- Tests Docker installation
- Verifies Docker daemon
- Tests build process
- Creates required directories

### Makefile Targets

#### Quick Start
```bash
make docker-deploy    # Build + start (first time)
make docker-quick     # Start with cache (faster)
```

#### Container Management
```bash
make docker-up        # Start containers
make docker-down      # Stop containers
make docker-restart   # Restart containers
make docker-ps        # Show container status
```

#### Debugging
```bash
make docker-logs      # View logs (Ctrl+C to exit)
make docker-shell     # Open bash in container
```

#### Cleanup
```bash
make docker-clean     # Remove all Docker resources
```

### Raw Docker Compose Commands

```bash
# Build image
docker compose build

# Start container (detached)
docker compose up -d

# Stop container
docker compose down

# View logs
docker compose logs -f cosmic-pipeline

# Container status
docker compose ps

# Execute command in container
docker compose exec cosmic-pipeline bash

# Rebuild without cache
docker compose build --no-cache

# Clean up everything
docker compose down --rmi all --volumes --remove-orphans
```

---

## Volume Mounts

### Why Volumes?

Model weights and data files are NOT committed to Git. They are mounted from your local filesystem into the container.

### Mounted Directories

| Local Path | Container Path | Purpose |
|------------|----------------|---------|
| `./models` | `/app/models` | LSTM model weights (`lstm_ae.pt`) |
| `./data/cache` | `/app/data/cache` | GOES API cache |
| `./data/raw` | `/app/data/raw` | CSV data files |
| `./config` | `/app/config` | YAML configuration |

### Important Notes

1. **Model Training**: Run `make train` or `python models/train.py` BEFORE Docker deployment
2. **Data Persistence**: All data in mounted volumes persists after container stops
3. **Hot Reload**: Changes to mounted files are immediately visible in container

---

## Troubleshooting

### Container Won't Start

```bash
# Check Docker daemon
docker ps

# Check logs
docker compose logs cosmic-pipeline

# Rebuild from scratch
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Port 8501 Already in Use

```bash
# Find process using port
netstat -ano | findstr :8501    # Windows
lsof -i :8501                   # Mac/Linux

# Kill process or change port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Model Not Found

```bash
# Train model locally first
python models/train.py

# Verify model exists
ls models/lstm_ae.pt    # Linux/Mac
dir models\lstm_ae.pt   # Windows

# Restart container
docker compose restart
```

### Permission Errors (Linux)

```bash
# Fix ownership
sudo chown -R $USER:$USER data/ models/

# Or run with sudo
sudo docker compose up -d
```

### Container Keeps Restarting

```bash
# Check health status
docker compose ps

# View detailed logs
docker compose logs --tail=100 cosmic-pipeline

# Check healthcheck
docker inspect cosmic-pipeline-app | grep -A 10 Health
```

---

## Performance Tips

### 1. Use Cache for Faster Builds
```bash
# First time (slow)
make docker-deploy

# Subsequent times (fast)
make docker-quick
```

### 2. Prune Unused Resources
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

### 3. Limit Resource Usage

Edit `docker-compose.yml`:
```yaml
services:
  cosmic-pipeline:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

---

## Development Workflow

### 1. Local Development
```bash
# Work on code locally
python dashboard/app.py

# Test changes
pytest tests/
```

### 2. Test in Docker
```bash
# Build and test
make docker-deploy

# View logs
make docker-logs

# Debug in container
make docker-shell
```

### 3. Production Deployment
```bash
# Clean build
make docker-clean
make docker-deploy

# Verify health
make docker-ps
```

---

## Security Notes

### 1. Environment Variables
Never commit secrets to Git. Use `.env` file:

```bash
# .env (not committed)
GOES_API_KEY=your_secret_key
```

Update `docker-compose.yml`:
```yaml
services:
  cosmic-pipeline:
    env_file:
      - .env
```

### 2. Network Isolation
Container runs on isolated bridge network `cosmic-net`.

### 3. Read-Only Mounts
For production, use read-only mounts:
```yaml
volumes:
  - ./config:/app/config:ro
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker compose build
      - name: Test container
        run: |
          docker compose up -d
          sleep 10
          curl -f http://localhost:8501/_stcore/health
          docker compose down
```

---

## FAQ

**Q: Do I need to rebuild after code changes?**
A: Yes, for Python code changes. No, for mounted files (data, config).

**Q: Can I run multiple instances?**
A: Yes, change the port mapping in `docker-compose.yml`.

**Q: How do I update dependencies?**
A: Update `requirements.txt`, then `docker compose build --no-cache`.

**Q: Where are logs stored?**
A: View with `docker compose logs`. Not persisted by default.

**Q: Can I use GPU in Docker?**
A: Yes, requires NVIDIA Docker runtime. See: https://github.com/NVIDIA/nvidia-docker

---

## Support

- **Issues**: https://github.com/omercangumus/-cosmic-pipeline/issues
- **Docker Docs**: https://docs.docker.com/
- **Streamlit Docs**: https://docs.streamlit.io/

---

**TUA Astro Hackathon 2026** | Ömer & Ahmet
