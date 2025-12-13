#!/bin/bash

set -e

echo "======================================"
echo "🔧 Movie Recommendation App - Fix Deployment"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Stop all containers...${NC}"
docker-compose down -v 2>/dev/null || true

echo -e "${YELLOW}Step 2: Remove dangling images...${NC}"
docker image prune -a -f --filter "until=1h" 2>/dev/null || true

echo -e "${YELLOW}Step 3: Create .env file...${NC}"
cat > .env << EOF
MONGO_USER=movieadmin
MONGO_PASSWORD=Mayank@03
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
EOF

echo -e "${GREEN}✓ .env file created${NC}"
cat .env

echo -e "${YELLOW}Step 4: Pull latest images...${NC}"
docker-compose pull

echo -e "${YELLOW}Step 5: Build Docker image...${NC}"
docker-compose build --no-cache

echo -e "${YELLOW}Step 6: Start containers...${NC}"
docker-compose up -d

echo -e "${YELLOW}Step 7: Waiting for services to be ready (30s)...${NC}"
sleep 30

echo -e "${YELLOW}Step 8: Check container status...${NC}"
docker-compose ps

echo -e "${YELLOW}Step 9: Check logs...${NC}"
docker-compose logs --tail=50

echo -e "${GREEN}======================================"
echo "✅ Deployment fixed successfully!"
echo "======================================${NC}"

echo -e "${YELLOW}Testing connectivity...${NC}"
curl -v http://localhost:8501 || echo "Streamlit may still be initializing..."

echo -e "${GREEN}Access your app at: http://$(hostname -I | awk '{print $1}'):8501${NC}"
