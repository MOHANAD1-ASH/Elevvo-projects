# Movie Recommendation System

A hybrid recommendation system built on the **MovieLens 100K** dataset, combining Content-Based Filtering and Collaborative Filtering to deliver personalized movie suggestions.

---

## Overview

| | |
|---|---|
| **Task** | Movie Recommendation |
| **Dataset** | MovieLens 100K — auto-downloaded at runtime |
| **Ratings** | 100,000 |
| **Users** | 943 |
| **Movies** | 1,682 |
| **Rating Scale** | 1 – 5 |
| **Approaches** | Content-Based Filtering · Collaborative Filtering |

---

## Dataset

The dataset is downloaded and extracted automatically at runtime from the GroupLens server — no manual setup required.

| File | Content |
|---|---|
| `u.data` | `user_id` · `movie_id` · `rating` · `timestamp` |
| `u.item` | Movie title · release date · 19 binary genre columns |

> `video_release_date`, `imdb_url`, and `unknown` columns were dropped due to missing values or irrelevance.

---

## Pipeline

```
Data Loading → EDA → Rating Analysis → Bayesian Averaging
→ Genre Matrix → Cosine Similarity → KNN Collaborative Filtering → Recommendations
```

---

## Methodology

### 1. Data Loading
Downloads and extracts `ml-100k.zip` from GroupLens, then loads ratings and movie metadata into separate DataFrames.

### 2. Exploratory Data Analysis
- Descriptive statistics for both ratings and movies
- Missing value and duplicate checks
- Rating distribution (count plot with mean and median)
- Distribution of ratings per user and per movie
- Matrix sparsity calculation

### 3. Rating Analysis
- Most popular movies ranked by rating count and average score
- Identification of highest and lowest rated movies by raw mean
- **Problem found:** the top-rated movie (*Great Day in Harlem*) had only a single rating — making the raw mean unreliable

### 4. Bayesian Average Rating

To correct the bias introduced by movies with very few ratings, a Bayesian average is applied:

$$r_i = \frac{C \times m + \sum{\text{reviews}}}{C + N}$$

| Symbol | Meaning |
|---|---|
| `C` | Average number of ratings per movie (confidence prior) |
| `m` | Global mean rating across all movies |
| `N` | Number of ratings for movie `i` |

**Result after correction:**
- Highest rated → *Star Wars (1977)*
- Lowest rated → *Leave It to Beaver (1997)*

### 5. Content-Based Filtering
Builds a genre feature matrix from 19 binary genre columns and computes pairwise **Cosine Similarity** between all movies. Recommendations are ranked by similarity score to the input title.

### 6. Collaborative Filtering
Constructs a sparse user–movie rating matrix (`csr_matrix`) and applies **K-Nearest Neighbors** on movie vectors. Recommendations are based on movies rated similarly by users who liked the input title.

### 7. Fuzzy Title Matching
A `movie_finder()` utility handles typos and partial titles using `fuzzywuzzy`, ensuring robust input handling before any recommendation is made.

---

## Usage

```python
# Content-Based — recommends by genre similarity
get_content_based_recommendations("Mission: Impossible", n_recommendations=10)

# Collaborative — recommends by user behavior
get_collaborative_recommendations("Mission: Impossible", n_recommendations=10)
```

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy fuzzywuzzy python-Levenshtein
```

---

## Running the Notebook

1. Open `elevvo3.ipynb` in Jupyter or Google Colab
2. Run all cells top to bottom — the dataset downloads automatically
