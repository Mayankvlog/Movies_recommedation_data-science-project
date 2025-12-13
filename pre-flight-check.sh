#!/bin/bash

set -e

echo "======================================"
echo "📋 Pre-flight Deployment Check"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 exists"
    else
        echo -e "${RED}✗${NC} $1 MISSING"
        ERRORS=$((ERRORS + 1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1 directory exists"
    else
        echo -e "${RED}✗${NC} $1 directory MISSING"
        ERRORS=$((ERRORS + 1))
    fi
}

echo ""
echo -e "${YELLOW}Checking required files...${NC}"
check_file ".env"
check_file "docker-compose.yml"
check_file "Dockerfile"
check_file "apps.py"
check_file "recommender.py"
check_file "requirements.txt"

echo ""
echo -e "${YELLOW}Checking required directories...${NC}"
check_dir "model"
check_dir "data"

echo ""
echo -e "${YELLOW}Checking model files...${NC}"
check_file "model/movie_recommender.h5"
check_file "model/tfidf_vectorizer.pkl"
check_file "model/movies_df.pkl"

echo ""
echo -e "${YELLOW}Checking data files...${NC}"
check_file "data/tmdb_5000_movies.csv"

echo ""
echo -e "${YELLOW}System Information:${NC}"
echo "Docker version:"
docker --version || echo "  Docker not found"

echo "Docker Compose version:"
docker-compose --version || echo "  Docker Compose not found"

echo ""
echo -e "${YELLOW}Environment Variables:${NC}"
if [ -f ".env" ]; then
    cat .env | sed 's/=.*/=***/' || true
else
    echo "  .env file not found"
fi

echo ""
echo -e "${YELLOW}Docker Images:${NC}"
docker images | grep -E "mongo|movies-recommendation" || echo "  No relevant images found"

echo ""
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}======================================"
    echo "✅ All checks passed!"
    echo "=====================================${NC}"
    exit 0
else
    echo -e "${RED}======================================"
    echo "❌ $ERRORS error(s) found!"
    echo "=====================================${NC}"
    exit 1
fi
