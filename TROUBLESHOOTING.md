# Movie Recommendation App - Troubleshooting Guide

## Connection Timeout Error

If you see `The connection has timed out` when accessing `http://your-ip:8501`, follow these steps:

### Quick Fix (Run on VPS)

```bash
cd ~/Movies_recommedation_data-science-project

# Pull latest fixes
git pull origin main

# Run the automated fix script
bash fix-deployment.sh

# Monitor the logs
docker-compose logs -f web
```

---

## Step-by-Step Debugging

### 1. Check Container Status

```bash
docker-compose ps
```

**Expected output:**
- `movie_recommendation_app` - UP (or status healthy)
- `movie_mongodb` - UP (or status healthy)

### 2. Check If Port is Open

```bash
curl http://localhost:8501
# or from another machine
curl http://YOUR_VPS_IP:8501
```

### 3. View Real-time Logs

```bash
# Streamlit app logs
docker-compose logs -f web

# MongoDB logs
docker-compose logs -f mongo

# All logs
docker-compose logs -f
```

### 4. Check Model Files Inside Container

```bash
docker-compose exec web bash

# Inside container, check if model files exist:
ls -la model/
ls -la data/

# Check Python environment
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import streamlit; print(streamlit.__version__)"

# Exit container
exit
```

### 5. Verify MongoDB Connectivity

```bash
docker-compose exec mongo mongosh

# Inside MongoDB shell:
db.adminCommand('ping')
exit
```

### 6. Test with Pre-flight Check

```bash
bash pre-flight-check.sh
```

---

## Common Issues & Solutions

### Issue: "Model files not found"

**Solution:**
```bash
docker-compose exec web ls -la model/
docker-compose exec web ls -la data/

# If missing, copy from your local machine or DVC
# First, ensure DVC files are tracked:
git status
```

**If DVC files are missing:**
```bash
# On VPS, pull data
dvc pull
```

### Issue: "TensorFlow import error"

**Solution:**
```bash
docker-compose down -v
docker image rm mayank035/movies-recommendation:latest
docker-compose pull
docker-compose build --no-cache
docker-compose up -d
```

### Issue: "Connection refused" or "Cannot connect to MongoDB"

**Solution:**
```bash
# Check if MongoDB is running
docker-compose exec mongo mongosh

# If that fails, restart MongoDB
docker-compose restart mongo

# Wait 10 seconds then check health
sleep 10
docker-compose ps
```

### Issue: "Out of memory" or container keeps restarting

**Solution:**
```bash
# Check memory usage
docker stats

# If using limited server, adjust docker-compose.yml:
# Reduce memory limits for web service (current: 2G)
# Edit docker-compose.yml and reduce resources.limits.memory
# Then restart:
docker-compose down -v
docker-compose up -d
```

### Issue: "Streamlit app loading but blank page"

**Solution:**
```bash
# Clear Streamlit cache
docker-compose exec web rm -rf ~/.streamlit

# Restart
docker-compose restart web

# Monitor logs
docker-compose logs -f web
```

---

## Full Restart Procedure

If nothing else works, do a complete fresh start:

```bash
# 1. Stop everything
docker-compose down -v

# 2. Clean up Docker
docker system prune -a -f

# 3. Pull fresh images
docker-compose pull

# 4. Recreate .env
cat > .env << EOF
MONGO_USER=movieadmin
MONGO_PASSWORD=Mayank@03
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
EOF

# 5. Start fresh
docker-compose up -d

# 6. Wait and monitor
sleep 30
docker-compose logs -f
```

---

## Performance Monitoring

### Check Resource Usage
```bash
docker stats --no-stream
```

### Check Disk Space
```bash
df -h
docker system df
```

### Check Network
```bash
netstat -tlnp | grep 8501
netstat -tlnp | grep 27017
```

---

## Useful Commands

| Command | Purpose |
|---------|---------|
| `docker-compose ps` | Show container status |
| `docker-compose logs -f web` | Watch Streamlit logs |
| `docker-compose exec web bash` | Shell into web container |
| `docker-compose restart web` | Restart Streamlit |
| `docker-compose down` | Stop all containers |
| `docker system prune -a -f` | Clean up Docker |
| `docker stats` | Monitor resource usage |

---

## Checking Streamlit Startup in Detail

```bash
docker-compose up -d
sleep 60
docker-compose logs web | tail -100
```

**Look for these success messages:**
```
✓ TensorFlow imported successfully
✓ Recommender module imported successfully
✓ Streamlit page config set
✓ All artifacts loaded successfully
✓ Embeddings generated: shape (5000, 256)
✓ ALL SYSTEMS GO - APP READY
```

If you don't see these messages, the app is likely crashing on startup.

---

## Environment Variables to Verify

Inside container:
```bash
docker-compose exec web bash -c "env | grep -E 'MONGO|STREAMLIT|PYTHON'"
```

Should show:
```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
PYTHONUNBUFFERED=1
MONGO_URI=mongodb://movieadmin:Mayank@03@mongo:27017/moviesdb
```

---

## GitHub Actions Deployment

Check workflow status on GitHub:
```
https://github.com/Mayankvlog/Movies_recommedation_data-science-project/actions
```

If workflow fails:
1. Check GitHub Actions logs
2. Verify VPS_HOST, VPS_USER, VPS_PASSWORD secrets are correct
3. Test SSH connection manually first:
   ```bash
   ssh -i your_key root@VPS_IP
   ```

---

## Need Help?

If issues persist after following this guide:

1. **Collect diagnostic info:**
```bash
echo "=== Container Status ===" && docker-compose ps
echo "=== Recent Logs ===" && docker-compose logs --tail=200
echo "=== Disk Space ===" && df -h
echo "=== Memory Usage ===" && free -h
```

2. **Share the output when asking for help**

3. **Check if model files exist on VPS:**
```bash
ls -la model/
ls -la data/
```

---

**Last Updated:** December 13, 2025
**App Version:** Latest
**Status:** ✅ Production Ready
