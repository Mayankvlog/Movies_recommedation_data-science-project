# MongoDB Integration Guide

## Overview
MongoDB has been successfully integrated into your Movie Recommendation System for data persistence and tracking.

## Collections Created

### 1. **movies** Collection
Stores all movie data with the following fields:
- `_id`: MongoDB ObjectId (auto-generated)
- `id`: Original movie ID
- `title`: Movie title
- `genres`: Genre information
- `overview`: Movie description
- `popularity`: Popularity score

**Usage**: Reference data for recommendations

### 2. **training_logs** Collection
Tracks training metrics for model optimization:
- `timestamp`: When the metric was logged
- `epoch`: Training epoch number
- `train_loss`: Training loss value
- `val_loss`: Validation loss value
- `model_name`: Name of the model being trained

**Usage**: Monitor model performance over time

### 3. **recommendations** Collection
Stores all user recommendation queries:
- `timestamp`: When the recommendation was made
- `selected_movie`: The movie the user selected
- `recommendations`: List of recommended movies
- `source`: Source of recommendation (e.g., "streamlit_app")
- `user_session_id`: Unique session identifier for tracking user behavior

**Usage**: Analytics and user behavior tracking

## Configuration

### Environment Variable
Set the MongoDB connection URI via environment variable:
```bash
export MONGO_URI="mongodb://localhost:27017/moviesdb"
```

**Default**: `mongodb://localhost:27017/moviesdb`

### Docker Setup
If using Docker, add to `docker-compose.yml`:
```yaml
services:
  mongodb:
    image: mongo:latest
    container_name: moviesdb
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_DATABASE: moviesdb
    volumes:
      - mongo_data:/data/db
volumes:
  mongo_data:
```

## Usage in Code

### In `movie_system.ipynb`

1. **Initialize MongoDB**:
```python
mongo_client, mongo_db = init_mongodb()
```

2. **Save Movies to Database**:
```python
save_movies_to_db(movies_df, mongo_db)
```

3. **Log Training Metrics**:
```python
log_training_metrics(mongo_db, {
    "train_loss": train_loss,
    "val_loss": val_loss,
    "model_name": "autoencoder"
}, epoch=epoch)
```

### In `apps.py` (Streamlit App)

1. **Get MongoDB Client**:
```python
mongo_client, mongo_db = get_mongo_client()
```

2. **Save Recommendations**:
```python
save_recommendation_to_db(mongo_db, selected_movie, recommendations)
```

3. **Retrieve Recommendation History**:
```python
history = get_recommendation_history(mongo_db, limit=20)
```

4. **Get Training Metrics**:
```python
metrics = get_training_metrics(mongo_db)
```

## Features in Streamlit App

### Tab 1: Get Recommendations
- Select a movie and get recommendations
- Recommendations are automatically saved to MongoDB
- Shows timestamp and session tracking

### Tab 2: Recommendation History
- View the last 20 recommendations made
- Download history as CSV for analysis
- Tracks all queries with timestamps

### Tab 3: Model Info
- Display total movies in database
- Show TF-IDF feature count
- Display embedding dimensions
- Show total recommendations made
- Display latest training metrics

## Benefits

✅ **Data Persistence**: All recommendations are saved for future analysis
✅ **Training Monitoring**: Track model performance across training runs
✅ **User Analytics**: Understand user behavior and popular movies
✅ **Scalability**: MongoDB scales with your dataset
✅ **Query Flexibility**: Easy querying and filtering of historical data
✅ **CSV Export**: Download recommendation data for external analysis

## Troubleshooting

### MongoDB Connection Error
**Error**: `Failed to connect to MongoDB`
- **Solution**: Ensure MongoDB is running on your machine or Docker container
- **Check**: Run `mongosh` command or verify MongoDB service status

### Collection Not Found
**Error**: Collection doesn't exist
- **Solution**: This is normal on first run. Collections are auto-created when data is first saved.

### Permission Denied
**Error**: Authentication failed
- **Solution**: Verify credentials in MONGO_URI environment variable

## Dependencies
- `pymongo==4.7.1` (Already in requirements.txt)

## Next Steps
1. Start MongoDB locally or via Docker
2. Run the Jupyter notebook to train and save data
3. Launch the Streamlit app
4. View recommendations being saved in real-time
5. Check recommendation history in the app's History tab
