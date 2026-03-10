# 🎬 Movie Recommendation System

A **content-based Movie Recommendation System** built using **Python** and **Machine Learning** techniques.
The system recommends movies similar to a user-selected movie by analyzing movie metadata such as **overview, genres, keywords, cast, and crew**.

---

## 📌 Project Overview

This project uses the **TMDB 5000 Movie Dataset** to build a recommendation engine that suggests movies based on similarity.

The recommendation model works by combining multiple movie features into a single text representation and converting them into numerical vectors. Using **cosine similarity**, the system finds movies that are most similar to the selected one.

---

## ⚙️ Technologies Used

* Python
* Pandas
* Scikit-learn
* CountVectorizer
* Cosine Similarity
* Streamlit (for web interface)

---
### File Description

**app.py**
Creates the web interface using Streamlit where users can select a movie and get recommendations.

**recommender.py**
Handles the machine learning logic including:

* dataset preprocessing
* feature extraction
* vectorization
* similarity calculation
* recommendation generation

**datasets/**
Contains the TMDB movie dataset used to train the recommendation model.

---

## 🧠 How the Recommendation System Works

1. Load movie and credits datasets.
2. Merge the datasets based on the movie title.
3. Select important features such as:

   * overview
   * genres
   * keywords
   * cast
   * crew
4. Combine these features into a single column called **tags**.
5. Convert the tags into vectors using **CountVectorizer**.
6. Compute similarity between movies using **Cosine Similarity**.
7. When a user selects a movie, the system finds the **top 5 most similar movies**.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/your-username/movie-recommendation-system.git
```

### 2. Navigate to the Project Folder

```
cd movie-recommendation-system
```

### 3. Install Required Libraries

```
pip install -r requirements.txt
```

### 4. Run the Application

If using Streamlit:

```
streamlit run app.py
```

Or run the recommendation script directly:

```
python recommender.py
```

---

## 🎥 Example Output

```
Enter a movie name: Avatar

Recommended Movies:

Guardians of the Galaxy
John Carter
Star Trek
Aliens
The Fifth Element
```

---

## 📊 Dataset

The dataset used in this project is the **TMDB 5000 Movie Dataset**, which contains movie metadata including genres, cast, crew, and plot summaries.

---

## 📈 Future Improvements

Possible improvements for the project include:

* Adding movie posters using the TMDB API
* Deploying the application online
* Using **TF-IDF Vectorization** for better recommendations
* Implementing a **hybrid recommendation system**

---

## 👨‍💻 Author

**Rajat Chandra Ghimire**

Computer Science Student
Passionate about **Machine Learning, Python, and AI projects**.
