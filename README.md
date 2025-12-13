# 🎬 Movie Recommendation System - Data Science Project

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/streamlit-app-FF69B4.svg)](https://streamlit.io/)

A production-ready, end-to-end machine learning movie recommendation system powered by deep learning. This project demonstrates complete data science practices including model training with MLflow tracking, automated testing, containerization, and a user-friendly web interface.

## 🚀 Overview

This intelligent movie recommendation engine analyzes movie genres using a 5-layer neural network autoencoder architecture. The system learns compressed representations of movies through TF-IDF vectorization and generates personalized recommendations using cosine similarity metrics on learned embeddings.

**Key Capabilities:**
- Real-time movie recommendations based on user selection
- Scalable deep learning architecture with multiple activation functions
- Complete experiment tracking and versioning with MLflow
- Automated testing with comprehensive test suite
- Docker containerization for easy deployment
- MongoDB integration for data persistence

## ✨ Key Features

### 🧠 Deep Learning Architecture
- **5-Layer Autoencoder**: Advanced neural network with multiple activation functions
  - Layer 1: 512 neurons (ReLU activation)
  - Layer 2: 256 neurons (Tanh activation)
  - Layer 3: 128 neurons (ELU activation)
  - Layer 4: 64 neurons (SELU activation)
  - Embedding Layer: 32 neurons (Sigmoid activation)
- **Adaptive Learning Rate**: Adam optimizer with 0.001 learning rate
- **Loss Function**: Mean Squared Error for reconstruction

### 📊 MLOps & Experiment Tracking
- **MLflow Integration**: Complete experiment tracking and model versioning
- **DagsHub Integration**: Remote experiment management and collaboration
- **Data Version Control**: DVC support for data and model versioning
- **Model Serialization**: H5 format for trained models and pickle for preprocessing artifacts

### 🎨 Interactive Web Application
- **Streamlit Interface**: Beautiful, responsive web UI
- **Real-time Predictions**: Instant recommendations on movie selection
- **Error Handling**: Graceful handling of missing dependencies
- **Custom Styling**: Modern gradient-based design

### ✅ Testing & Quality Assurance
- **PyTest Integration**: Comprehensive automated test suite
- **Unit Tests**: Model loading and recommendation accuracy validation
- **CI/CD Ready**: GitHub Actions integration for automated testing

### 🐳 Deployment & Containerization
- **Docker Support**: Complete containerization setup
- **Docker Compose**: Multi-container orchestration
- **Easy Deployment**: Production-ready deployment configuration

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Machine Learning** | TensorFlow, Keras, Scikit-learn |
| **MLOps** | MLflow, DagsHub, DVC, PyTest |
| **Frontend** | Streamlit, Altair |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Database** | MongoDB |
| **Deployment** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |

## 📋 Prerequisites

- **Python 3.11+**
- **Git** (for version control)
- **MongoDB** (optional, for production deployment)
- **Docker & Docker Compose** (for containerized deployment)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mayankvlog/Movies_recommedation_data-science-project.git
cd Movies_recommedation_data-science-project
```

### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃 Quick Start

### Option 1: Run Streamlit Web App (Recommended)
```bash
# Activate virtual environment
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Launch Streamlit app
streamlit run apps.py
```
The app will open at `http://localhost:8501`

### Option 2: Train Model Directly
```bash
# Run the Jupyter notebook
jupyter notebook movie_system.ipynb
```

### Option 3: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the app at http://localhost:8501
```

## 📁 Project Structure

```
Movies_recommedation_data-science-project/
├── 📊 data/
│   ├── tmdb_5000_movies.csv          # TMDB dataset
│   └── tmdb_5000_movies.csv.dvc      # Data versioning
├── 🤖 model/
│   ├── movie_recommender.h5          # Trained autoencoder
│   ├── movie_recommender.h5.dvc      # Model versioning
│   ├── tfidf_vectorizer.pkl.dvc      # TF-IDF vectorizer
│   └── movies_df.pkl.dvc             # Preprocessed dataframe
├── 🧪 test/
│   ├── test_model.py                 # Model validation tests
│   └── __pycache__/
├── 📱 apps.py                         # Streamlit application
├── 🔧 recommender.py                  # Recommendation engine
├── 📓 movie_system.ipynb              # Training notebook
├── 🐳 Dockerfile                      # Docker image config
├── 🐳 docker-compose.yml              # Docker Compose config
├── 📦 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
└── 📚 myenv/                          # Virtual environment
```

## 🔄 Workflow & Pipeline

### 1. Data Loading & Preprocessing (Steps 1-3)
- Load TMDB 5000 movies dataset
- Extract genre information from JSON format
- Create text soup for vectorization

### 2. Feature Engineering (Step 4)
- Apply TF-IDF vectorization (1000 features)
- Create genre-based feature vectors
- Train-validation split (80-20)

### 3. Model Training (Steps 5-7)
- Build 5-layer autoencoder architecture
- Compile with Adam optimizer
- Train for 50 epochs with batch size 32
- Log metrics to MLflow for tracking

### 4. Model Persistence (Step 8)
- Save trained encoder model
- Serialize TF-IDF vectorizer
- Save preprocessed dataframe

## 🎯 How It Works

### Recommendation Algorithm
1. **User Input**: User selects a movie from the dropdown
2. **Encoding**: Movie genres encoded to TF-IDF vector
3. **Embedding**: Vector passed through encoder to get 32-D embedding
4. **Similarity Calculation**: Compute cosine similarity with all movies
5. **Ranking**: Return top 5 most similar movies

### Model Architecture Flow
```
Input (1000 features)
    ↓
Dense(512, ReLU)
    ↓
Dense(256, Tanh)
    ↓
Dense(128, ELU)
    ↓
Dense(64, SELU)
    ↓
Dense(32, Sigmoid) ← Embedding Layer
    ↓
Dense(64, SELU)
    ↓
Dense(128, ELU)
    ↓
Dense(256, Tanh)
    ↓
Dense(512, ReLU)
    ↓
Dense(1000, Softmax)
    ↓
Output (1000 features - Reconstruction)
```

## 🧪 Testing

Run the automated test suite:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test/test_model.py
```

### Test Coverage
- ✅ Model loading validation
- ✅ Data preprocessing verification
- ✅ Recommendation accuracy checks
- ✅ Error handling scenarios

## 📊 Experiment Tracking

### MLflow Logging
The training pipeline automatically logs:
- **Parameters**: Epochs (50), batch size (32), TF-IDF features (1000)
- **Metrics**: Training loss, validation loss (per epoch)
- **Artifacts**: Training loss plot, encoder model
- **Models**: Encoder model in MLflow registry

### View Experiments
```bash
# Launch MLflow UI
mlflow ui
```
Access at `http://localhost:5000`

## 🔗 DagsHub Integration

Connect to remote experiment management:
```python
# Already configured in notebook
dagshub.init(
    repo_owner='Mayankvlog',
    repo_name='Movies_recommedation_data-science-project',
    mlflow=True
)
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t movie-recommender .
```

### Run Container
```bash
# Development mode
docker run -p 8501:8501 movie-recommender

# Production with Docker Compose
docker-compose up -d
```

### Configuration
- **Port**: 8501 (Streamlit default)
- **Volume Mounts**: Model and data directories
- **Environment**: Python 3.11

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Encoding Dimension** | 32 |
| **Training Epochs** | 50 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 |
| **TF-IDF Features** | 1000 |
| **Train-Val Split** | 80-20 |

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: MongoDB Connection Error
**Solution**: MongoDB is optional. App works without it using CSV data.

### Issue: Port 8501 already in use
**Solution**: Run on different port
```bash
streamlit run apps.py --server.port 8502
```

### Issue: CUDA/GPU not found
**Solution**: CPU mode is automatic. For GPU support:
```bash
pip install tensorflow-gpu
```

## 📝 Data Source

- **Dataset**: [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Records**: 5000 movies
- **Features**: Title, genres, release date, and more

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Mayank Vlog**
- GitHub: [@Mayankvlog](https://github.com/Mayankvlog)
- Repository: [Movies_recommedation_data-science-project](https://github.com/Mayankvlog/Movies_recommedation_data-science-project)

## 🙏 Acknowledgments

- TMDB for the movie dataset
- TensorFlow & Keras team
- Streamlit for the amazing framework
- MLflow for experiment tracking
- DagsHub for collaboration platform

## 📧 Support & Contact

For issues, questions, or suggestions:
- 🐛 Open an [GitHub Issue](https://github.com/Mayankvlog/Movies_recommedation_data-science-project/issues)
- 💬 Start a [Discussion](https://github.com/Mayankvlog/Movies_recommedation_data-science-project/discussions)

## 🚀 Future Enhancements

- [ ] User ratings integration
- [ ] Collaborative filtering approach
- [ ] Content-based + collaborative hybrid model
- [ ] Real-time feedback loop
- [ ] Advanced visualization dashboards
- [ ] API deployment with FastAPI
- [ ] Multi-language support
- [ ] Mobile application

---

**Last Updated**: December 2025  
**Project Status**: Active Development





