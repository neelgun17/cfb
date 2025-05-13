# College Football QB Archetype Project

**Version:** 1.0 (Date: YYYY-MM-DD)
**Author(s):** [Your Name/Group]

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Goals](#goals)
3.  [Data Source & Features](#data-source--features)
    *   [Initial Data](#initial-data)
    *   [Feature Engineering & Selection](#feature-engineering--selection)
4.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Dimensionality Reduction](#dimensionality-reduction)
    *   [Clustering Algorithm](#clustering-algorithm)
    *   [Manual Adjustments](#manual-adjustments)
5.  [QB Archetype Definitions (k=4, Hierarchical Clustering)](#qb-archetype-definitions)
    *   [Archetype 1: [Scrambling Survivors]](#archetype-1)
    *   [Archetype 2: [Pocket Managers]](#archetype-2)
    *   [Archetype 3: [Dynamic Dual-Threats]](#archetype-3)
    *   [Archetype 4: [Mobile Pocket Passers]](#archetype-4)
6.  [Key Findings & Observations](#key-findings--observations)
7.  [Limitations](#limitations)
8.  [Future Work & Potential Applications](#future-work--potential-applications)
9.  [Setup & Usage (If Applicable)](#setup--usage)
10. [Contributing (If Applicable)](#contributing)
11. [License (If Applicable)](#license)

---

## 1. Project Overview
This project aims to analyze 2024 projected PFF college football quarterback data to identify distinct player archetypes. By leveraging nearly 300 statistical columns, the goal is to group QBs based on their on-field tendencies, efficiencies, and play styles. These archetypes can then be used for applications such as predicting system fit for transfers, scouting, and further machine learning endeavors.

---

## 2. Goals
*   To process and refine a large dataset of PFF QB statistics.
*   To identify natural groupings (archetypes) of quarterbacks using unsupervised machine learning techniques.
*   To define these archetypes based on their statistical profiles and representative players.
*   To create a labeled dataset that can be used for future supervised learning tasks.
*   Ultimately, to develop a system for recommending player-team fits based on QB archetype and offensive scheme profiles.

---

## 3. Data Source & Features

### 3.1. Initial Data
*   **Source:** Pro Football Focus (PFF) College Football Data
*   **Season:** 2024 (Projected Stats)
*   **Initial Columns:** Approximately 300 columns covering a wide range of passing, rushing, grading, and situational statistics for quarterbacks.
*   **Initial Player Pool:** [514] QBs after initial load.
*   **Filtering:** Players with fewer than [Your Dropback Threshold, e.g., 125] dropbacks were excluded to ensure statistical stability, resulting in a final pool of [153] QBs for clustering.

### 3.2. Feature Engineering & Selection
The initial ~400 features were reduced to a more manageable and impactful set of approximately [Number, e.g., 38] features for clustering. The selection process prioritized:
*   Rate and percentage-based statistics over raw counts to normalize for playing time.
*   Key PFF grades (Overall, Passing, Rushing).
*   Metrics capturing rushing *impact* and *tendency* (e.g., `designed_run_rate`, `scramble_rate`, `ypa_rushing`, `elusive_rating`, `percent_total_yards_rushing`, `qb_rush_attempt_rate`).
*   Overall passing efficiency and style indicators (e.g., `accuracy_percent`, `avg_depth_of_target`, `btt_rate`, `twp_rate`, `ypa`).
*   Key situational tendency and performance metrics (e.g., `pa_rate`, `pa_ypa`, `deep_attempt_rate`, `deep_accuracy_percent`, `pressure_rate`, `pressure_sack_percent`).
*   A conscious effort was made to balance the number of passing-focused vs. rushing-focused features to prevent one aspect from unduly dominating the clustering.

*(You can list the exact `features_for_clustering_REVISED` here or link to a separate file if it's too long for the README).*

---

## 4. Methodology

### 4.1. Data Preprocessing
1.  **Handling Missing Values:** Missing numerical data in the selected features was imputed using the median value for each respective column.
2.  **Feature Scaling:** All selected features were scaled using `sklearn.preprocessing.StandardScaler` to ensure each feature contributed appropriately to the distance calculations in the clustering algorithm.

### 4.2. Dimensionality Reduction
*   **Method:** [PCA]
*   **Parameters:** [e.g., For PCA: `n_components` set to retain 90% of variance.]
*   **Result:** The [301] scaled features were reduced to [33] dimensions for clustering.

### 4.3. Clustering Algorithm
*   **Algorithm:** Hierarchical Agglomerative Clustering (`sklearn.cluster.AgglomerativeClustering`)
*   **Linkage Method:** 'ward' (minimizes the variance of the clusters being merged)
*   **Number of Clusters (k):** 4 (determined by analyzing dendrograms, silhouette scores from previous K-Means iterations, and interpretability of resulting cluster profiles).

### 4.4. Manual Adjustments
Following the algorithmic clustering, a manual review of player assignments was conducted. For a small number of players whose statistical profiles clearly indicated a better fit with a different archetype than assigned by the algorithm, manual reclassifications were made. These include:
*   [Player Name 1]: Moved from [Algorithmic Cluster X] to [Final Archetype Y] because [Brief Justification based on key stats].
*   [Player Name 2]: Moved from [...] to [...] because [...].
*   *(List all manual changes here for transparency)*

---

## 5. QB Archetype Definitions (k=4, Hierarchical Clustering - Finalized)

Below are the definitions of the four QB archetypes identified, based on the mean statistical profiles of players within each cluster (after any manual adjustments).

*(This is where you'll insert the detailed descriptions based on the final stats and player lists you provide me after your review. I'll use placeholders for now based on our latest discussion for the HIERARCHICAL k=4 run)*

### 5.1. Archetype: ["Scrambling Survivors"]
*   **Cluster ID (Hierarchical):** 0
*   **Core Identity:** [e.g., Quarterbacks who frequently extend plays through scrambling, often under pressure, with a more reactive than proactive run game. Their passing can be risky and less efficient than other archetypes.]
*   **Key Statistical Indicators (Mean Values from `hierarchical_profiles_k4.csv` for this cluster):**
    *   `designed_run_rate`: [Value]
    *   `scramble_rate`: [Value - Highest of the 4]
    *   `ypa_rushing`: [Value]
    *   `elusive_rating`: [Value]
    *   `avg_depth_of_target`: [Value]
    *   `accuracy_perc`: [Value]
    *   `twp_rate`: [Value - Likely high]
    *   `btt_rate`: [Value - Likely low to moderate]
    *   `pressure_rate`: [Value - Highest]
    *   `pressure_sack_percent`: [Value - Highest]
    *   `grades_pass`: [Value - Likely lower end]
*   **Representative Players:**
    *   [Player A from this cluster]
    *   [Player B from this cluster]
    *   [Player C from this cluster]
    *   [Player D from this_cluster]
    *   [Player E from this cluster]

### 5.2. Archetype: ["Pocket Managers"]
*   **Cluster ID (Hierarchical):** 1
*   **Core Identity:** [e.g., Traditional passers with limited designed mobility or scrambling effectiveness. Their game is built on operating from the pocket, often with a shorter, controlled passing attack and varying degrees of efficiency.]
*   **Key Statistical Indicators (Mean Values):**
    *   `designed_run_rate`: [Value - Lowest]
    *   `scramble_rate`: [Value - Lowest]
    *   `ypa_rushing`: [Value - Lowest]
    *   `elusive_rating`: [Value - Lowest]
    *   `avg_depth_of_target`: [Value - Lowest]
    *   `accuracy_perc`: [Value - Good to Very Good]
    *   `twp_rate`: [Value - Moderate to Low]
    *   `grades_pass`: [Value - Good]
*   **Representative Players:**
    *   [Player A from this cluster - e.g., Kyle McCord, Noah Fifita (after manual move)]
    *   [Player B from this cluster - e.g., Carson Beck]
    *   [Player C from this cluster - e.g., Shedeur Sanders (if moved here)]
    *   [Player D from this cluster - e.g., Cam Ward (if moved here)]
    *   [Player E from this cluster]

### 5.3. Archetype: ["Dynamic Dual-Threats"]
*   **Cluster ID (Hierarchical):** 2
*   **Core Identity:** [e.g., Quarterbacks who are significant threats both through the air and on the ground, often with high designed run involvement and the ability to make explosive plays. Their passing can be aggressive and effective, especially with play-action.]
*   **Key Statistical Indicators (Mean Values):**
    *   `designed_run_rate`: [Value - Highest]
    *   `scramble_rate`: [Value - High]
    *   `ypa_rushing`: [Value - Highest]
    *   `elusive_rating`: [Value - Highest]
    *   `grades_run`: [Value - High]
    *   `avg_depth_of_target`: [Value - High]
    *   `btt_rate`: [Value - High]
    *   `accuracy_perc`: [Value - Good]
    *   `twp_rate`: [Value - Moderate (higher than HC3)]
    *   `pa_rate` / `pa_ypa`: [Values - High]
*   **Representative Players:**
    *   [Player A from this cluster - e.g., Jalen Milroe]
    *   [Player B from this cluster - e.g., Riley Leonard]
    *   [Player C from this cluster - e.g., Jaxson Dart]
    *   [Player D from this cluster]
    *   [Player E from this cluster]

### 5.4. Archetype: ["Mobile Pocket Passers"]
*   **Cluster ID (Hierarchical):** 3
*   **Core Identity:** [e.g., Quarterbacks who exhibit top-tier passing efficiency, accuracy, and ball security. While not primary designed runners, they possess enough pocket mobility and scrambling ability to extend plays. Their game is defined by elite passing command.]
*   **Key Statistical Indicators (Mean Values):**
    *   `designed_run_rate`: [Value - Moderate/Low]
    *   `scramble_rate`: [Value - Moderate]
    *   `elusive_rating`: [Value - Low/Moderate]
    *   `accuracy_perc`: [Value - Highest]
    *   `twp_rate`: [Value - Lowest]
    *   `btt_rate`: [Value - High]
    *   `grades_pass`: [Value - Highest]
    *   `grades_offense`: [Value - Highest]
*   **Representative Players:**
    *   [Player A from this cluster - e.g., If Sanders/Ward *stayed* here, they'd be listed. If not, who defines this cluster now? Perhaps the Klubniks, Allars who pass very well and have some designed mobility but aren't the dominant rushers of HC2.]
    *   [Player B from this cluster]
    *   [Player C from this cluster]
    *   [Player D from this cluster]
    *   [Player E from this cluster]

---

## 6. Key Findings & Observations
*   The hierarchical clustering approach with a revised, balanced feature set yielded four statistically distinct QB archetypes.
*   A key success was the clearer separation of "Dynamic Dual-Threats" (HC2) from "Elite Efficient Passers (with some pocket mobility)" (HC3) and "Pocket Managers" (HC1), which was a challenge with earlier K-Means iterations.
*   Players with elite passing efficiency but minimal designed run games (e.g., [Sanders, Ward if you moved them to HC1]) were better classified by [explaining your final decision/manual adjustment rationale].
*   Noah Fifita presented a unique challenge, initially misclassified by [K-Means/Hierarchical HC0] due to [likely low BTT rate], but his overall profile strongly aligns with [HC1 - Pocket Managers] after manual review.
*   The "Scrambling Survivors" (HC0) capture a group that relies heavily on out-of-structure plays and faces significant pressure, often leading to riskier passing outcomes.
*   Low-volume players were generally distributed among the archetypes based on their limited statistical profiles.

---

## 7. Limitations
*   **Unsupervised Nature:** Clustering identifies patterns but doesn't guarantee perfect alignment with preconceived notions or all individual player nuances. Some "edge case" players may not fit perfectly into any single archetype.
*   **Feature Sensitivity:** The choice and weighting of features can significantly influence clustering outcomes. While efforts were made to balance the feature set, different selections could yield different groupings.
*   **"Hard" Clustering:** Both K-Means (initially explored) and Hierarchical Clustering (with a fixed k) assign each player to only one cluster, which may not fully capture hybrid play styles.
*   **Impact of Dimensionality Reduction:** PCA, while necessary, involves some information loss which can affect clustering.

---

## 8. Future Work & Potential Applications
*   **System Fit Analysis:**
    *   Develop statistical profiles for common college offensive schemes.
    *   Create a matching algorithm or scoring system to recommend QB-team fits based on archetype alignment.
*   **Supervised Classification Model:**
    *   Train a classifier (e.g., RandomForest) on the finalized, labeled archetypes to automate future archetype assignments and gain insights from feature importances.
*   **Exploration of Other Clustering Algorithms:** Investigate Gaussian Mixture Models (GMM) for "soft" probabilistic assignments to better handle hybrid players.
*   **Temporal Analysis:** If multi-year data becomes available, track how QB archetypes evolve over a player's career or when changing teams/schemes.
*   **Integration of Film Study:** Combine statistical archetyping with qualitative film review for a more holistic player evaluation.
*   **Refine "Low Volume" Handling:** Develop a more robust strategy for players who barely meet the minimum play_count/dropback threshold, as their rate stats can be volatile.

---

## 9. Setup & Usage (If Applicable)
*(If you plan to share your code, describe how to set it up and run it.)*
*   **Dependencies:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`.
*   **Running the Code:**
    *   `python main_script.py` (or whatever your main script is called)
    *   Ensure `your_qb_data.csv` is in the specified path.
*   **Outputs:**
    *   `hierarchical_profiles_k4.csv`: Mean statistical profiles of the 4 archetypes.
    *   `hierarchical_player_assignments_k4.csv`: Player list with assigned numerical and named archetypes.
    *   Dendrogram plot.
