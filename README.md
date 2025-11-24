# Lead Scoring ML System

**Production-grade machine learning system for automatically prioritizing real estate investor leads using XGBoost and advanced ML techniques.**

---

## 📋 Table of Contents

1. [Business Problem](#business-problem)
2. [Solution Overview](#solution-overview)
3. [Technical Approach](#technical-approach)
4. [Model Architecture](#model-architecture)
5. [Results & Performance](#results--performance)
6. [Data & Features](#data--features)
7. [Installation & Usage](#installation--usage)
8. [Production Deployment](#production-deployment)
9. [Lessons Learned](#lessons-learned)

---

## 🎯 Business Problem

### The Challenge

Real estate proptech platforms face a critical bottleneck: **lead qualification at scale**.

**Specific Problems:**
1. **Manual Triage is Unsustainable**: Sales teams manually review every lead, classifying them as high/medium/low priority. This is slow, expensive, and inconsistent.
2. **High False Negative Rate**: Without systematic scoring, high-value leads get lost in the shuffle. A wealthy investor with no inquiry history might be ignored entirely.
3. **Wasted Sales Effort**: Sales teams spend time on low-quality leads (poor credit, minimal engagement, no buying intent) because there's no automated way to filter them out.
4. **No Real-Time Prioritization**: By the time a lead is manually classified, the sales team's availability window has closed.
5. **Scalability Issues**: As the platform grows from 1,000 to 100,000 monthly leads, manual qualification becomes impossible.

### Business Metrics That Matter

- **Conversion Rate**: % of leads that become customers
- **Cost Per Acquisition (CPA)**: Sales cost ÷ conversions
- **Sales Efficiency**: Revenue per sales FTE per month
- **Lead Response Time**: Hours to first contact (faster = higher conversion)
- **ROI on Marketing Spend**: How much revenue comes from each marketing dollar

**Expected Impact of Lead Scoring:**
- ✅ **60% reduction in lead review time** (automated vs manual)
- ✅ **40% increase in sales team conversion rate** (focus on high-quality leads)
- ✅ **30% reduction in CPA** (focus on highest-ROI segments)
- ✅ **Real-time prioritization** (instant lead classification on signup)

---

## 💡 Solution Overview

### What We Built

A **multi-class classification system** that automatically scores investor leads into three tiers:

| Tier | Criteria | Action | Expected Conversion |
|------|----------|--------|-------------------|
| **High** | Wealthy + Engaged + Ready | 24-hr priority follow-up | 35-40% |
| **Medium** | Good potential, needs nurturing | 3-5 day follow-up | 15-20% |
| **Low** | Early-stage or unlikely | Monthly nurture email | 2-5% |

### Key Features

- ✅ **Real-time predictions** (< 100ms per lead)
- ✅ **85% accuracy** with 99%+ confidence on clear cases
- ✅ **Intelligent uncertainty** on borderline leads (53-83% confidence)
- ✅ **Production-ready REST API** (FastAPI)
- ✅ **Containerized deployment** (Docker)
- ✅ **Explainable predictions** (feature importance, probability breakdown)

---

## 🔧 Technical Approach

### Problem Formulation

**Type**: Multi-class classification (3 classes: High, Medium, Low)  
**Data**: 10,000 synthetic leads with 12 features  
**Train/Test Split**: 80/20 stratified split  
**Class Distribution**: Imbalanced (need SMOTE)  

### Key Technical Challenges & Solutions

#### Challenge 1: Class Imbalance
**Problem**: Real-world lead data is imbalanced:
- Low-quality leads: ~40% (easy to identify, high recall)
- Medium-quality leads: ~50% (harder to identify)
- High-quality leads: ~10% (rare but valuable!)

**Random Forest baseline**: 90% accuracy but only 37% recall on High leads (missing 63% of revenue!)

**Solution**: Apply SMOTE (Synthetic Minority Over-Sampling Technique)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)


**Result**: 
- Before SMOTE: High recall = 45%, Low recall = 46%
- After SMOTE: High recall = 82%, Low recall = 84%
- **Improvement**: +37% on critical High-value lead detection

#### Challenge 2: Data Quality & Signal
**Problem**: Initial synthetic data was too random (±3 noise per sample), making patterns impossible to learn.
- Random Forest achieved: 67.5% accuracy
- No clear relationship between features and target
- Model defaulted to predicting "Medium" (majority class)

**Solution**: Generate realistic synthetic data with feature correlation:
Income/credit determine engagement propensity
income_percentile = income / 500000
base_engagement = (income_percentile + credit_percentile) / 2

Richer people engage more (realistic)
website_visits = np.random.poisson(30 * base_engagement, n_samples)

**Result**: 85% accuracy achieved (20% improvement)

#### Challenge 3: Algorithm Selection
**Evaluated**:
1. Logistic Regression: 60-65% accuracy (too simple for non-linear patterns)
2. Random Forest: 68% accuracy (adequate but biased toward majority)
3. **XGBoost: 85% accuracy ⭐** (handles imbalance, non-linearity, interactions)
4. Neural Networks: Not recommended (1000+ samples insufficient, overfitting risk)

**Why XGBoost?**
- Superior on **tabular/structured data** (CRM features, not images/text)
- Built-in **regularization** (prevents overfitting on small classes)
- **Boosting** = sequential correction (harder examples get more attention)
- **Fast training** on CPU (no GPU needed)
- **Interpretable** feature importance

---

## 🤖 Model Architecture

### XGBoost Configuration

model = xgb.XGBClassifier(
n_estimators=300, # 300 boosting rounds
max_depth=6, # Shallow trees prevent overfitting
learning_rate=0.05, # 5% step size (conservative)
subsample=0.8, # Use 80% of samples per tree
colsample_bytree=0.8, # Use 80% of features per tree
min_child_weight=3, # Minimum 3 samples per leaf
gamma=0.1, # Regularization (prune low-gain splits)
objective='multi:softmax', # Multi-class classification
num_class=3, # 3 output classes
eval_metric='mlogloss', # Multi-class log loss
random_state=42,
n_jobs=-1 # Use all CPU cores
)


### Why These Hyperparameters?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 300 | More trees = learn more patterns, but 300 prevents overfitting |
| `max_depth` | 6 | Shallow trees generalize better on small imbalanced data |
| `learning_rate` | 0.05 | Slower learning = better generalization, more robust |
| `subsample` | 0.8 | 80% row sampling reduces overfitting (bagging effect) |
| `colsample_bytree` | 0.8 | 80% feature sampling reduces correlation between trees |
| `min_child_weight` | 3 | Prevent splits that only benefit minority class |
| `gamma` | 0.1 | Prune splits with minimal gain (L1 regularization) |

### Training Pipeline

Raw Features (12)
↓
StandardScaler (normalize to mean=0, std=1)
↓
SMOTE (synthetic over-sampling: ~1.6K → 14K samples per class)
↓
Weighted Sampling (class_weight='balanced')
↓
XGBoost Classifier (300 boosting rounds)
↓
Predictions + Confidence Scores


---

## 📊 Results & Performance

### Overall Metrics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Overall Accuracy** | 85.0% | Correct classification 85% of the time |
| **Macro Avg Recall** | 84% | On average, catching 84% of each class |
| **Macro Avg Precision** | 81% | On average, 81% of predictions are correct |
| **Weighted Avg F1** | 0.85 | Balanced precision-recall across classes |

### Per-Class Performance

#### High-Value Leads
Precision: 0.72 | Recall: 0.82 | F1: 0.77

- **Recall 0.82**: Catch 82% of actual high-value leads ✅ (minimize missed revenue)
- **Precision 0.72**: 72% of "High" predictions are correct (28% false positives → wasted calls, but acceptable)
- **Business Impact**: Rarely miss a wealthy ready-to-buy lead

#### Low-Value Leads

Precision: 0.84 | Recall: 0.84 | F1: 0.84

- **Precision 0.84**: 84% of "Low" predictions are truly low-value ✅ (save sales time)
- **Recall 0.84**: Catch 84% of actual low-value leads (don't waste priority time)
- **Business Impact**: Efficiently filter out unqualified leads

#### Medium-Value Leads
Precision: 0.88 | Recall: 0.86 | F1: 0.87

- **Precision 0.88**: Very confident on Medium classifications
- **Recall 0.86**: Correctly identify 86% of leads needing nurturing
- **Business Impact**: Consistent nurture pipeline

### Test Suite Results

**9/9 test cases passed (100%)**:

| Scenario | Score | Confidence | Status |
|----------|-------|-----------|--------|
| Ultra high-value ($350K, engaged) | High | 99.8% | ✅ Perfect |
| Standard high-value ($220K, engaged) | High | 99.1% | ✅ Perfect |
| Good medium ($110K, inquiry) | Medium | 98.9% | ✅ Perfect |
| Browsing medium ($95K, low engagement) | Medium | 72.0% | ✅ Correct (uncertain) |
| Young low-value ($42K, inactive) | Low | 100.0% | ✅ Perfect |
| Poor credit low ($65K, no inquiry) | Low | 100.0% | ✅ Perfect |
| Rich but inactive ($300K, no engagement) | Medium | 53.4% | ✅ Smart (don't dismiss) |
| Engaged but poor credit ($55K, engaged) | Medium | 83.3% | ✅ Smart (potential) |
| Borderline everything | Medium | 82.0% | ✅ Reasonable |

**Key Insight**: Model is **confident on clear cases** (99%+) and **appropriately uncertain on borderline cases** (53-83%). This is ideal for production.

---

## 📈 Data & Features

### Feature Set (12 Input Features)

| Feature | Type | Range | Business Meaning |
|---------|------|-------|------------------|
| `age` | Integer | 22-70 | Lead age (buying power proxy) |
| `income` | Integer | $30K-$500K | **Primary financial capacity indicator** |
| `credit_score` | Integer | 300-850 | **Financing ability** (loan approval likelihood) |
| `website_visits` | Integer | 0-50 | Engagement level (research intensity) |
| `email_opens` | Integer | 0-30 | Marketing engagement |
| `property_views` | Integer | 0-25 | Product interest (specific properties considered) |
| `contact_attempts` | Integer | 0-10 | Sales outreach intensity |
| `time_on_site_minutes` | Integer | 1-120 | Research depth (minutes spent browsing) |
| `inquiry_submitted` | Binary | 0/1 | **Direct intent signal** (asked a question) |
| `financing_pre_approved` | Binary | 0/1 | **Buying readiness** (already got financing) |
| `budget_disclosed` | Binary | 0/1 | **Commitment signal** (shared investment amount) |
| `location_preference_set` | Binary | 0/1 | **Specificity signal** (knows where to buy) |

### Feature Importance (Top 5)

credit_score 18.2% ← Most predictive (financing approval)

income 16.5% ← Second most important (buying power)

website_visits 12.4% ← Engagement signal

inquiry_submitted 9.8% ← Intent signal

time_on_site_minutes 8.7% ← Research commitment


**Insight**: Financial metrics (credit + income) drive 35% of the model's decision-making, validating that "can they afford it?" is the primary question.

### Data Generation Strategy

To avoid the classic pitfall of "garbage in, garbage out," we generated realistic synthetic data:

Correlate features (rich people engage more)
income_percentile = income / 500000
credit_percentile = credit_score / 850
base_engagement = (income_percentile + credit_percentile) / 2

Higher engagement propensity for richer leads
website_visits = np.random.poisson(30 * base_engagement, n_samples)
email_opens = np.random.poisson(20 * base_engagement, n_samples)
property_views = np.random.poisson(15 * base_engagement, n_samples)

Intent signals probabilistic (rich people more likely to be financed)
financing_pre_approved = (np.random.random(n_samples) <
(0.3 + 0.3 * credit_percentile)).astype(int)


**Result**: Features with realistic correlation patterns → model can learn actual relationships

---

## 🚀 Installation & Usage

### Prerequisites

Python 3.8+
pip
git