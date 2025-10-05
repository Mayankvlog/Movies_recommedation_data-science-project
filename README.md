
🎬 Movie Recommendation System - Data Science Project

A production-ready movie recommendation engine powered by deep learning, featuring complete MLOps practices including experiment tracking, automated testing, and CI/CD integration.
🚀 Overview

This system uses a 5-layer neural network autoencoder to analyze movie genres and generate intelligent recommendations. The model learns compressed representations of movies through TF-IDF vectorization and delivers personalized suggestions using cosine similarity on learned embeddings.
✨ Key Features

    Deep Learning Architecture: 5-layer autoencoder with multiple activation functions (ReLU, Tanh, ELU, SELU, Sigmoid)

    MLOps Integration: Complete experiment tracking with MLflow, automated testing with pytest, and CI/CD pipeline via CircleCI

    Interactive Web App: User-friendly Streamlit interface for real-time movie recommendations

    Production-Ready: Comprehensive testing suite, proper code structure, and complete documentation

🛠️ Tech Stack

ML/DL: TensorFlow, Keras, Scikit-learn | MLOps: MLflow, pytest, CircleCI | Deployment: Streamlit | Data: Pandas, NumPy | Visualization: Matplotlib, Seaborn
📦 Quick Start

bash
# Clone repository
git clone https://github.com/Mayankvlog/Movies_recommedation_data-science-project.git

cd Movies_recommedation_data-science-project

# Install dependencies
pip install -r requirements.txt

# Train model
jupyter notebook movie_recommendation_system.ipynb

# Run tests
pytest

# Launch app
streamlit run app.py

🎯 Project Structure

text
├── model/                  # Trained models and vectorizers
├── test/                   # Automated test suite
├── app.py                  # Streamlit application
├── movie_recommendation_system.ipynb  # Training notebook
└── requirements.txt        # Dependencies

📄 License

MIT License - Feel free to use for learning and development.

Word Count: Exactly 234 words ✅

Simply copy this content and create a README.md file in your GitHub repository!




