#!/bin/bash

set -e

echo "======================================"
echo "🔧 Movie Recommendation App - Full Deployment Fix"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Running pre-flight checks...${NC}"
bash pre-flight-check.sh || true

echo ""
echo -e "${YELLOW}Step 1: Stop all containers and clean up...${NC}"
docker-compose down -v 2>/dev/null || true
docker system prune -f --filter "until=24h" 2>/dev/null || true

echo ""
echo -e "${YELLOW}Step 2: Remove dangling images...${NC}"
docker image prune -a -f --filter "until=1h" 2>/dev/null || true

echo ""
echo -e "${YELLOW}Step 3: Create .env file...${NC}"
cat > .env << EOF
MONGO_USER=movieadmin
MONGO_PASSWORD=Mayank@03
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
EOF

echo -e "${GREEN}✓ .env file created${NC}"
echo "Contents:"
cat .env

echo ""
echo -e "${YELLOW}Step 4: Pull latest images...${NC}"
docker-compose pull || true

echo ""
echo -e "${YELLOW}Step 5: Build Docker image (if needed)...${NC}"
docker-compose build --no-cache || docker-compose build

echo ""
echo -e "${YELLOW}Step 6: Start containers with detailed logging...${NC}"
docker-compose up -d

echo ""
echo -e "${YELLOW}Step 7: Wait for services to be ready...${NC}"
echo "Waiting 10 seconds..."
sleep 10

echo "Checking MongoDB health..."
for i in {1..15}; do
    if docker exec movie_mongodb mongosh --eval "db.adminCommand('ping')" &>/dev/null; then
        echo -e "${GREEN}✓ MongoDB is healthy${NC}"
        break
    fi
    echo "  Attempt $i/15..."
    sleep 2
done

echo ""
echo -e "${YELLOW}Step 8: Check container status...${NC}"
docker-compose ps

echo ""
echo -e "${YELLOW}Step 9: Check resource usage...${NC}"
docker stats --no-stream

echo ""
echo -e "${YELLOW}Step 10: View recent logs (web container)...${NC}"
echo "================================"
docker-compose logs --tail=100 web
echo "================================"

echo ""
echo -e "${YELLOW}Step 11: View recent logs (mongo container)...${NC}"
echo "================================"
docker-compose logs --tail=50 mongo
echo "================================"

echo ""
echo -e "${GREEN}======================================"
echo "✅ Deployment fixed and started!"
echo "======================================${NC}"

echo ""
echo -e "${BLUE}Useful commands for troubleshooting:${NC}"
echo "  docker-compose ps                    # Check container status"
echo "  docker-compose logs -f web           # Watch Streamlit logs"
echo "  docker-compose logs -f mongo         # Watch MongoDB logs"
echo "  docker-compose exec web bash         # Shell into web container"
echo "  curl http://localhost:8501           # Test Streamlit (if local)"
echo ""

IP=$(hostname -I | awk '{print $1}')
echo -e "${GREEN}Access your app at:${NC}"
echo "  http://$IP:8501"
echo ""

echo -e "${YELLOW}Checking connectivity (may take a few seconds)...${NC}"
for i in {1..10}; do
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        echo -e "${GREEN}✓ Streamlit is responding!${NC}"
        break
    fi
    echo "  Attempt $i/10..."
    sleep 2
done

