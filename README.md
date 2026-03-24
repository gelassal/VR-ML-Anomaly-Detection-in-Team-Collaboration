# VR-ML-Anomaly-Detection-in-Team-Collaboration
Developed an ML pipeline to detect teamwork breakdowns in VR environments using Isolation Forest. Transformed raw behavioral data into meaningful features to identify anomalies in coordination without requiring labeled datasets.



## Overview
This project explores how to automatically detect breakdowns in teamwork within Mixed Reality (MR) environments using unsupervised machine learning.

We analyze multimodal interaction data (movement, attention, and coordination) from collaborative VR sessions and apply an Isolation Forest model to identify anomalous behavior that corresponds to poor collaboration.

The system produces a **group-level anomaly intensity score over time**, enabling detection of sustained breakdowns in team coordination without requiring labeled data.

---

## Motivation
Mixed Reality collaboration generates rich behavioral data, but evaluating teamwork performance is difficult because:
- Data is noisy and high-dimensional  
- Collaboration breakdowns are emergent (not individual)  
- Labeled datasets are expensive or unavailable  

This project addresses these challenges using **unsupervised anomaly detection** to identify deviations from normal team behavior.

---

## Key Idea
Instead of labeling “good vs bad teamwork,” we:
1. Learn what *normal collaboration* looks like  
2. Detect deviations using anomaly detection  
3. Interpret anomalies as **teamwork breakdowns**

---

## System Pipeline

### 1. Data Collection
- Mixed Reality collaborative sessions
- Pairwise (dyadic) interaction data per second
- Captures:
  - Spatial relationships (distance, movement)
  - Shared attention
  - Interaction dynamics

---

### 2. Feature Engineering
Selected features representing coordination and engagement:
- `dist_accel` — relative movement acceleration  
- `dist_mean` — average distance between teammates  
- `approach_rate` — rate of movement toward/away  
- `shared_att_ratio` — shared attention alignment  
- `joint_att_count` — joint attention events  

Unstable or noisy features were removed to improve model reliability.

---

### 3. Model: Isolation Forest
- Unsupervised anomaly detection
- Works by randomly partitioning data using isolation trees
- Anomalies:
  - Easier to isolate → shorter path lengths  
- Normal points:
  - Dense → longer paths  

Each data point is assigned:
- An anomaly score  
- A binary label (anomalous / normal)

---

### 4. Group-Level Anomaly Metric
Raw data is dyadic (pairwise), so we:

- Compute anomaly labels per pair per second  
- Aggregate across all pairs  
- Generate a **group-level anomaly intensity (0–1)**  

> 1 = fully anomalous collaboration  
> 0 = normal collaboration  

---

### 5. Temporal Breakdown Detection
Instead of single spikes, we detect:

- **Sustained anomaly streaks (>50% intensity)**  
- Allow short gaps (≤5 seconds) due to noise  

These streaks represent:
- meaningful breakdowns in teamwork  
- potential intervention points  

---

## Evaluation & Validation

### Data Cleaning
- Removed corrupted sessions (e.g., idle recordings lasting ~22 hours)
- Prevented model bias toward inactivity

---

### Model Stability
- Ran 100 iterations with different random seeds
- Observed consistent anomaly windows across runs

Confirms reliability of Isolation Forest predictions

---

### Behavioral Validation
- High anomaly intensity correlated with:
  - Longer task completion times  
  - Observed disengagement  

Example:
- High-anomaly group → ~2× longer completion time  

---

## Results

- Successfully identified sustained breakdown periods in collaboration  
- Group-level anomaly signal was:
  - More interpretable than individual-level signals  
  - More robust to noise  

- Demonstrated that:
  - **Data quality matters as much as model choice**
  - Simple models can outperform complex ones when designed properly  

---

## Key Insights

- Unsupervised learning works well for human behavior analysis  
- Group-level metrics are more meaningful than individual signals  
- Temporal streaks are better indicators than isolated anomalies  
- Data preprocessing is critical in real-world ML systems  

---

## Future Work

- Larger datasets for better generalization  
- Hybrid models with weak supervision  
- Real-time MR feedback systems:
  - Adaptive prompts  
  - Attention guidance  
  - Performance interventions  

---

## Tech Stack
- Python  
- scikit-learn (Isolation Forest)  
- Data preprocessing & feature engineering  
- Visualization tools (matplotlib / similar)  

---

## Contributors
- George Elassal  
- Mahdi Ayman  
- Jared Hrycak  
- Melinda Tran  

---

## References
Key concepts based on:
- Isolation Forest (Liu et al.)
- Anomaly detection in time-series data
- MR collaboration and proxemics research

M. Munir, M. A. Chattha, A. Dengel and S. Ahmed, "A Comparative Analysis of Traditional and Deep Learning-Based Anomaly Detection Methods for Streaming Data," 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, 2019, pp. 561-566, doi: 10.1109/ICMLA.2019.00105.

Liu, F. T., Ting, K. M., and Zhou, Z.-H. 2012. Isolation-based anomaly detection. ACM Trans. Knowl. Discov. Data 6, 1, Article 3 (March 2012), 39 pages. DOI = 10.1145/2133360.2133363 http://doi.acm.org/10.1145/2133360.2133363

Kim, I., & Sung, J. (2024). New proxemics in new space: proxemics in VR. Virtual Reality, 28, 85. https://doi.org/10.1007/s10055-024-00982-5.

Richardson, D. C., & Dale, R. (2005). Looking To Understand: The Coupling Between Speakers' and Listeners' Eye Movements and Its Relationship to Discourse Comprehension. Cognitive Science, 29(6), 1045–1060

Nidhi Ahlawat and Amit Awekar. 2024. Incremental Isolation Forest to Handle Concept Drift in Anomaly Detection. In 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD) (CODS-COMAD 2024), January 04–07, 2024, Bangalore, India. ACM, New York, NY, USA, 2 pages. https://doi.org/10.1145/3632410. 3632486


---

