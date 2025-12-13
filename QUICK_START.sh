#!/bin/bash

# ============================================
# Movie Recommendation App - Quick Commands
# ============================================
# Copy these commands to your VPS terminal

# 1. SSH into VPS
ssh root@139.59.56.109

# 2. Navigate to project
cd ~/Movies_recommedation_data-science-project

# 3. Get latest code with timeout fixes
git pull origin main

# 4. Run automated deployment fix
bash fix-deployment.sh

# 5. Or run manual commands:

# Stop everything
docker-compose down -v

# Create fresh .env
cat > .env << EOF
MONGO_USER=movieadmin
MONGO_PASSWORD=Mayank@03
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
EOF

# Pull and build
docker-compose pull
docker-compose build --no-cache

# Start services
docker-compose up -d

# Wait for services to be healthy
sleep 60

# Check status
docker-compose ps

# View logs
docker-compose logs -f web

# ============================================
# Monitoring Commands
# ============================================

# Check if app is responding (wait ~60s after startup)
curl http://localhost:8501

# View logs from last 100 lines
docker-compose logs --tail=100 web

# Real-time logs
docker-compose logs -f web

# Check resource usage
docker stats

# Check open ports
netstat -tlnp | grep -E "8501|27017"

# Shell into web container for debugging
docker-compose exec web bash

# ============================================
# Quick Restart
# ============================================

# If just one container crashes:
docker-compose restart web
docker-compose restart mongo

# If you need to rebuild the image:
docker-compose build --no-cache
docker-compose up -d

# Full clean restart:
docker-compose down -v
docker system prune -a -f
docker-compose up -d

# ============================================
# Accessing the App
# ============================================

# From VPS:
curl http://localhost:8501

# From your local computer:
# http://139.59.56.109:8501

# ============================================
# Key Files Updated
# ============================================
# 1. apps.py - Now has detailed logging
# 2. docker-compose.yml - Better health checks & resource limits
# 3. Dockerfile - Improved startup with health checks
# 4. fix-deployment.sh - Comprehensive deployment fix
# 5. pre-flight-check.sh - Validates setup before deployment
# 6. TROUBLESHOOTING.md - Full debugging guide

# ============================================
# Important Notes
# ============================================
# - Streamlit needs 60 seconds to load models on first start
# - MongoDB uses port 27017 (standard MongoDB port)
# - Streamlit UI runs on port 8501
# - Check logs with: docker-compose logs -f web
# - All environment variables are in .env file (gitignored)

echo "✅ Quick reference ready! Use the commands above."
