# College Football QB Archetype Project

**Version:** 1.0 
**Author(s):** Neel Gundlapally

## Table of Contents

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Goals](#goals)
3.  [Data Source & Features](#data-source--features)
    *   [Initial Data](#initial-data)
    *   [Feature Engineering & Selection](#feature-engineering--selection)
4.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Dimensionality Reduction (for Clustering Exploration)](#dimensionality-reduction-for-clustering-exploration) 
    *   [Archetype Discovery: Clustering Algorithm](#archetype-discovery-clustering-algorithm)
    *   [Manual Adjustments to Archetypes](#manual-adjustments-to-archetypes) <!-- MODIFIED -->
    *   **[NEW SECTION] Archetype Prediction: Supervised Classification(#archetype-prediction-supervised-classification)**
5.  [QB Archetype Definitions (k=4, Finalized)](#qb-archetype-definitions) 
    *   [Archetype 1: Scrambling Survivors](#archetype-1)
    *   [Archetype 2: Pocket Managers](#archetype-2)
    *   [Archetype 3: Dynamic Dual-Threats](#archetype-3)
    *   [Archetype 4: Mobile Pocket Passers](#archetype-4)
6.  [Key Findings & Results](#key-findings--results)
    *   [Clustering Insights](#clustering-insights) 
    *   [Classification Model Performance](#classification-model-performance)
7.  [Limitations](#limitations)
8.  [Future Work & Potential Applications](#future-work--potential-applications)
9.  [Setup & Usage (If Applicable)](#setup--usage)
10. [Contributing (If Applicable)](#contributing)
11. [License (If Applicable)](#license)

---

## 1. Project Overview
This project aims to analyze 2024 projected PFF college football quarterback data to identify distinct player archetypes. By leveraging nearly 300 statistical columns, the goal is to group QBs based on their on-field tendencies, efficiencies, and play styles. **Following the discovery of these archetypes through unsupervised clustering, a supervised classification model was developed to predict these archetypes for new players.** These archetypes and the predictive model can then be used for applications such as predicting system fit for transfers, scouting, and further machine learning endeavors.


---

## 2. Goals
*   To process and refine a large dataset of PFF QB statistics.
*   To identify natural groupings (archetypes) of quarterbacks using unsupervised machine learning techniques.
*   To define these archetypes based on their statistical profiles and representative players.
*   To develop an accurate supervised classification model to predict the identified QB archetypes based on player statistics.
*   To create a labeled dataset that can be used for future supervised learning tasks and to provide feature importances for archetype differentiation.
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

### 4.3. Archetype Discovery: Clustering Algorithm
*   **Algorithm:** Hierarchical Agglomerative Clustering (`sklearn.cluster.AgglomerativeClustering`)
*   **Linkage Method:** 'ward' (minimizes the variance of the clusters being merged)
*   **Number of Clusters (k):** 4 (determined by analyzing dendrograms, silhouette scores from previous K-Means iterations, and interpretability of resulting cluster profiles).
---
### 4.4. Archetype Prediction: Supervised Classification
Once the QB archetypes were finalized, a supervised learning approach was employed to build a model capable of predicting these archetypes from player statistics.
1.  **Data Preparation:**
    *   The finalized archetype labels were used as the target variable.
    *   The same set of [Number, e.g., 38] scaled features used for informing the clustering process was used as input features for the classifier.
    *   The data was split into training (75%) and testing (25%) sets, with stratification to maintain class proportions.
2.  **Model Selection & Tuning:**
    *   Several models were evaluated (Logistic Regression, SVC, default Random Forest, default XGBoost).
    *   **Random Forest Classifier** was selected as the best-performing base model.
    *   Hyperparameter tuning was conducted using `GridSearchCV` with 3-fold stratified cross-validation, optimizing for `f1_weighted` score.
    *   The `class_weight='balanced'` parameter was utilized to address the imbalance in archetype representation.
3.  **Best Model Parameters (Random Forest):**
    *   `class_weight`: 'balanced'
    *   `max_depth`: 5
    *   `min_samples_leaf`: 2
    *   `min_samples_split`: 10
    *   `n_estimators`: 300

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
### 6.1. Clustering Insights
*   The hierarchical clustering approach with a revised, balanced feature set yielded four statistically distinct QB archetypes.
*   A key success was the clearer separation of "Dynamic Dual-Threats" (HC2) from "Elite Efficient Passers (with some pocket mobility)" (HC3) and "Pocket Managers" (HC1), which was a challenge with earlier K-Means iterations.
*   Players with elite passing efficiency but minimal designed run games (e.g., [Sanders, Ward if you moved them to HC1]) were better classified by [explaining your final decision/manual adjustment rationale].
*   Noah Fifita presented a unique challenge, initially misclassified by [K-Means/Hierarchical HC0] due to [likely low BTT rate], but his overall profile strongly aligns with [HC1 - Pocket Managers] after manual review.
*   The "Scrambling Survivors" (HC0) capture a group that relies heavily on out-of-structure plays and faces significant pressure, often leading to riskier passing outcomes.
*   Low-volume players were generally distributed among the archetypes based on their limited statistical profiles.
*   The hierarchical clustering approach, using Ward's linkage on [Number, e.g., 38] scaled features, successfully identified four statistically distinct QB archetypes from the [153] qualifying players.
*   Key differentiators initially observed from cluster profiles included [mention 2-3 key differentiating aspects from your cluster profile analysis, e.g., "rushing volume/efficiency, passing depth/risk, and performance under pressure."]
*   Manual adjustments were minimal but crucial for refining assignments for a few players whose statistical nuances were better captured by domain knowledge.
*   A key success was the clearer separation of "Dynamic Dual-Threats" from "Mobile Pocket Passers" and "Pocket Managers."
*   The "Scrambling Survivors" capture a group that relies heavily on out-of-structure plays.
---
### 6.2. Classification Model Performance
A Random Forest classifier was trained and tuned to predict the four finalized QB archetypes. The model demonstrated strong performance on the held-out test set:

*   **Overall Test Accuracy:** **0.8718**
*   **Macro Averaged F1-Score (Test Set):** 0.87
*   **Weighted Averaged F1-Score (Test Set):** 0.87

**Per-Class Performance (Test Set):**

| Archetype              | Precision | Recall | F1-Score | Support |
| :--------------------- | :-------- | :----- | :------- | :------ |
| Dynamic Dual-Threats   | 0.83      | 1.00   | 0.91     | 5       |
| Mobile Pocket Passer   | 1.00      | 0.71   | 0.83     | 7       |
| Pocket Managers        | 0.85      | 0.94   | 0.89     | 18      |
| Scrambling Survivors   | 0.88      | 0.78   | 0.82     | 9       |
**Key Feature Importances (from Best Random Forest Classifier):**
The model found the following features most influential in distinguishing between archetypes:
1.  `grades_offense`
2.  `td_int_ratio`
3.  `completion_percent`
4.  `grades_pass`
5.  `pressure_rate`
    *(List your top 5-10, or refer to a plot)*

This high performance indicates that the statistically-derived archetypes are learnable and predictable from player features, validating their distinctiveness.

---

## 7. Limitations
*   **Unsupervised Nature of Archetype Definition:** Clustering identifies patterns but doesn't guarantee perfect alignment with preconceived notions. The "ground truth" labels for the classifier are derived from this unsupervised process (+ manual tweaks).
*   **Small Sample Size for "Dynamic Dual-Threats":** While the classifier performed surprisingly well on this class in the test set (N=5), the overall population for this archetype (N=18) is small, which can impact model robustness and generalization for this specific group.
*   **Feature Sensitivity:** The choice of features impacts both clustering and classification.
*   **"Hard" Assignments:** Both the clustering and classification assign players to a single archetype, which may not fully capture hybrid styles.
*   **Impact of PCA (for initial k-exploration):** While PCA was used for initial K-Means exploration of k, it was not used for the final hierarchical clustering or supervised model features, mitigating direct information loss in the final models. 

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
