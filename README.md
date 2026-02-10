# PulseLM: Towards Physiological Language Models

## Introduction

**PulseLM** is a multimodal framework that integrates PPG (Photoplethysmography) signal encoders with large language models for physiological signal understanding. The project includes a large-scale standardized PPG dataset and a model architecture that combines pretrained PPG encoders with LLM backbones (LLaMA, Qwen variants) via LoRA fine-tuning.

Each sample consists of:
- A **10-second PPG signal** (1,250 samples at 125 Hz, normalized to [0,1])
- A **text description** of the signal characteristics
- **Question-answer pairs** across 13 clinical categories with predefined answer choices

This dataset enables research in:
- Multimodal signal-language alignment
- Zero-shot physiological signal classification
- Explainable AI for cardiovascular health monitoring
- Foundation models for wearable biosignals

| Metric | Value |
|--------|-------|
| Total Datasets | 15 |
| Total Samples | ~1.31M |
| Signal Length | 1,250 samples (10 seconds) |
| Sampling Rate | 125 Hz |
| Signal Range | Normalized [0, 1] |
| QA Categories | 13 |
| Questions per Category | 10 |

---

## Model Architecture

PulseLM uses a multimodal architecture that fuses physiological signal features with text through an LLM backbone:

```
PPG Signal ──> PPG Encoder ──> Linear Projection ──┐
                                                    ├──> [PPG_tok, ECG_tok, Text_tok] ──> LLM (LoRA) ──> Answer
ECG Signal ──> ECG Encoder ──> Linear Projection ──┘
```

**Components:**
- **Signal Encoders**: Pretrained PPG and ECG encoders (frozen by default) that extract feature representations from raw signals
- **Projection Layers**: Linear layers that map encoder features to the LLM's hidden dimension
- **LLM Backbone**: Causal language model fine-tuned with LoRA on attention and MLP layers (`q/k/v/o/gate/up/down_proj`)
- **Supported LLMs**: LLaMA, Gemma, and Qwen variant families
- **Training Modes**: LoRA (default), full fine-tuning, or frozen LLM

---

## Setup

```bash
conda create -n ppg python=3.10 -y
conda activate ppg
pip install -r requirements.txt
```

---

## Datasets

### Summary

| Dataset | Samples | QA Categories | Source |
|---------|---------|---------------|--------|
| AFPPG | 4,196 | af_label | AF detection from PPG |
| AFPPGECG | 384,001 | af_label | AF detection from PPG+ECG |
| BCG | 671 | heart_rate, blood_pressure, sqi | BCG-PPG combined recordings |
| DALIA | 39,216 | heart_rate, activity | Daily life activities |
| Earset | 1,776 | heart_rate | Ear-worn PPG sensor |
| PPGArrhythmia | 46,827 | arrhythmia | Multi-class arrhythmia detection |
| PPGBP | 369 | heart_rate, blood_pressure | BP measurement study |
| SDB | 258,897 | sdb_label | Sleep polysomnography |
| Sensors | 2,061 | heart_rate, blood_pressure, sqi | Multi-sensor recordings |
| UCI | 111,751 | heart_rate, blood_pressure, sqi | UCI ML Repository |
| UQVitalSigns | 37,018 | heart_rate, blood_pressure, spo2, rr, sqi | Perioperative vital signs |
| UTSAPPG | 16,925 | heart_rate, hrv_sdnn, hrv_rmssd, hrv_pnn50 | UTSA PPG dataset |
| VitalDB | 163,959 | heart_rate, hrv_sdnn, hrv_rmssd, hrv_pnn50 | Surgical PPG recordings |
| WESAD | 2,998 | stress_label | Stress/Affect detection |
| WildPPG | 240,000 | heart_rate, hrv_sdnn, hrv_rmssd, hrv_pnn50 | Uncontrolled conditions |

---

## Signal Format

All signals are standardized to:

| Property | Value |
|----------|-------|
| Sampling Rate | 125 Hz |
| Duration | 10 seconds |
| Length | 1,250 samples |
| Normalization | Min-max [0, 1] |
| Data Type | float32 |

---

## QA Categories

### Overview

| Category | Answers | Datasets |
|----------|---------|----------|
| heart_rate_category | bradycardia, normal, tachycardia | BCG, DALIA, Earset, PPGBP, Sensors, UCI, UQVitalSigns, UTSAPPG, VitalDB, WildPPG |
| blood_pressure_category | normal, elevated, hypertension_stage1/2, crisis | BCG, PPGBP, Sensors, UCI, UQVitalSigns |
| sqi_category | good_quality, noisy_or_distorted, symmetric_unusual | BCG, Sensors, UCI, UQVitalSigns |
| activity_label | sitting, walking, cycling, stairs, driving, working, lunch_break, table_soccer, unknown | DALIA |
| stress_label | baseline, stress, amusement, meditation | WESAD |
| sdb_label | normal, mild, moderate, severe (AHI-based) | SDB |
| hrv_sdnn_category | low, normal, high | UTSAPPG, VitalDB, WildPPG |
| hrv_rmssd_category | low, normal, high | UTSAPPG, VitalDB, WildPPG |
| hrv_pnn50_category | low, normal, high | UTSAPPG, VitalDB, WildPPG |
| af_label | af, non_af | AFPPG, AFPPGECG |
| arrhythmia_category | sinus_rhythm, pvc, pac, vt, svt, af | PPGArrhythmia |
| spo2_category | normal, mild_hypoxemia, moderate_hypoxemia, severe_hypoxemia | UQVitalSigns |
| rr_category | bradypnea, normal, tachypnea | UQVitalSigns |

### Question Pools (10 questions per category)

<details>
<summary><b>Heart Rate Category</b></summary>

**Answers**: `bradycardia`, `normal`, `tachycardia`

1. What is the heart rate category for this PPG segment?
2. Classify the heart rate based on this waveform.
3. Which heart rate class does this sample belong to?
4. Is the heart rate normal, bradycardic, or tachycardic?
5. Provide the clinical heart rate category.
6. What heart rate classification does this PPG indicate?
7. Determine the heart rate category from the signal.
8. Based on the PPG waveform, what is the HR category?
9. Categorize the heart rate shown in this recording.
10. What is the heart rate status for this sample?
</details>

<details>
<summary><b>Blood Pressure Category</b></summary>

**Answers**: `normal`, `elevated`, `hypertension_stage1`, `hypertension_stage2`, `hypertensive_crisis`

1. What is the blood pressure category for this sample?
2. Classify the blood pressure level shown in this PPG segment.
3. Does this sample indicate normal blood pressure or hypertension?
4. Provide the blood pressure risk category.
5. What hypertension stage does this PPG correspond to?
6. Determine the BP classification from this waveform.
7. What is the blood pressure status for this recording?
8. Categorize the blood pressure level.
9. Based on the PPG, what is the BP category?
10. What blood pressure class does this sample belong to?
</details>

<details>
<summary><b>Signal Quality Index</b></summary>

**Answers**: `good_quality`, `noisy_or_distorted`, `symmetric_unusual`

1. Is this PPG signal clean or motion distorted?
2. How would you categorize the signal quality here?
3. Classify the PPG signal quality based on skewness.
4. Provide the SQI quality category for this sample.
5. What is the signal quality category for this PPG waveform?
6. Is this PPG recording of good or poor quality?
7. Determine the signal quality index category.
8. Rate the quality of this PPG signal.
9. What is the SQI classification for this segment?
10. Assess the signal quality of this PPG recording.
</details>

<details>
<summary><b>Activity Label</b></summary>

**Answers**: `sitting`, `walking`, `cycling`, `stairs`, `driving`, `working`, `lunch_break`, `table_soccer`, `unknown_0`

1. What is the activity label for this sample?
2. Classify the activity type.
3. Identify the physical activity for this segment.
4. What activity does this PPG correspond to?
5. What activity was being performed during this PPG recording?
6. Determine the activity category for this sample.
7. What type of activity is shown in this recording?
8. Categorize the physical activity.
9. What is the motion/activity state for this segment?
10. Identify what the subject was doing during this recording.
</details>

<details>
<summary><b>Stress Label</b></summary>

**Answers**: `baseline`, `stress`, `amusement`, `meditation`

1. What is the emotional state label?
2. What is the stress label for this segment?
3. Identify the stress level for this segment.
4. Provide the stress state for this PPG window.
5. What stress category does this sample belong to?
6. Determine the emotional/stress state.
7. What is the affective state for this recording?
8. Classify the stress level from this PPG.
9. What psychological state does this segment indicate?
10. Categorize the stress condition for this sample.
</details>

<details>
<summary><b>Sleep-Disordered Breathing</b></summary>

**Answers**: `normal_ahi<5`, `mild_5<=ahi<15`, `moderate_15<=ahi<30`, `severe_ahi>=30`

1. What is the breathing disorder category?
2. What is the sleep-disordered breathing label for this segment?
3. Classify the sleep breathing pattern.
4. Does this segment indicate sleep apnea?
5. Provide the SDB classification for this PPG window.
6. What is the AHI-based severity category?
7. Determine the sleep apnea severity.
8. Categorize the breathing disorder level.
9. What sleep-disordered breathing class is this?
10. Assess the respiratory disturbance category.
</details>

<details>
<summary><b>HRV SDNN Category</b></summary>

**Answers**: `low`, `normal`, `high`

1. What is the HRV SDNN category for this segment?
2. Classify the SDNN-based heart rate variability level.
3. How would you categorize SDNN for this PPG?
4. Is the SDNN low, normal, or high in this sample?
5. Provide the SDNN category based on this PPG segment.
6. What is the SDNN-based HRV classification?
7. Determine the SDNN level for this recording.
8. Categorize the overall HRV (SDNN) level.
9. What SDNN class does this sample belong to?
10. Assess the SDNN-based variability category.
</details>

<details>
<summary><b>HRV RMSSD Category</b></summary>

**Answers**: `low`, `normal`, `high`

1. What is the HRV RMSSD category for this segment?
2. Classify the RMSSD-based heart rate variability.
3. How would you categorize RMSSD here?
4. Is the RMSSD low, normal, or high?
5. Provide the RMSSD category for this sample.
6. What is the parasympathetic activity level (RMSSD)?
7. Determine the RMSSD classification.
8. Categorize the short-term HRV (RMSSD).
9. What RMSSD class does this recording indicate?
10. Assess the RMSSD-based variability level.
</details>

<details>
<summary><b>HRV pNN50 Category</b></summary>

**Answers**: `low`, `normal`, `high`

1. What is the pNN50 category for this segment?
2. Classify the pNN50 level.
3. How would you categorize pNN50 for this PPG?
4. Is pNN50 low, normal, or high in this sample?
5. Provide the pNN50 category.
6. What is the pNN50-based HRV classification?
7. Determine the pNN50 level for this recording.
8. Categorize the pNN50 variability measure.
9. What pNN50 class does this sample belong to?
10. Assess the pNN50 category from this PPG.
</details>

<details>
<summary><b>AF Label</b></summary>

**Answers**: `af`, `non_af`

1. Does this PPG signal show atrial fibrillation?
2. Is atrial fibrillation present in this recording?
3. Classify this PPG as AF or non-AF.
4. What is the AF detection result for this segment?
5. Determine whether this signal indicates atrial fibrillation.
6. Is this a normal rhythm or atrial fibrillation?
7. What is the AF label for this PPG recording?
8. Does this waveform indicate AF?
9. Provide the atrial fibrillation detection result.
10. Assess whether atrial fibrillation is present in this PPG.
</details>

<details>
<summary><b>Arrhythmia Category</b></summary>

**Answers**: `sinus_rhythm`, `pvc`, `pac`, `vt`, `svt`, `af`

1. What is the arrhythmia category for this segment?
2. Classify the cardiac rhythm in this recording.
3. What type of arrhythmia does this signal show?
4. Is this a normal rhythm or arrhythmia?
5. Determine the rhythm classification for this waveform.
6. What cardiac rhythm category does this sample belong to?
7. Identify the arrhythmia type from the signal.
8. Categorize the heart rhythm abnormality.
9. What is the rhythm diagnosis for this segment?
10. Assess the arrhythmia classification from this recording.
</details>

<details>
<summary><b>SpO2 Category</b></summary>

**Answers**: `normal`, `mild_hypoxemia`, `moderate_hypoxemia`, `severe_hypoxemia`

1. What is the SpO2 category for this segment?
2. Classify the blood oxygen saturation.
3. How would you categorize SpO2 for this PPG?
4. Is the SpO2 normal or does it indicate hypoxemia?
5. Provide the oxygen saturation category.
6. What is the oxygen saturation classification?
7. Determine the SpO2 category for this recording.
8. Categorize the SpO2 level.
9. What SpO2 class does this sample belong to?
10. Assess the oxygen saturation level from this PPG.
</details>

<details>
<summary><b>Respiratory Rate Category</b></summary>

**Answers**: `bradypnea`, `normal`, `tachypnea`

1. What is the respiratory rate category for this sample?
2. Classify the respiratory rate level.
3. How would you categorize the respiratory rate here?
4. Is the respiratory rate normal, slow, or fast?
5. Provide the respiratory rate category.
6. What is the breathing rate classification for this segment?
7. Determine the respiratory rate category for this recording.
8. Categorize the breathing rate.
9. What respiratory rate class does this sample belong to?
10. Assess the respiratory rate category from this PPG.
</details>

---

## Category Conversion Rules

### Heart Rate
| Value (bpm) | Label |
|-------------|-------|
| < 60 | `bradycardia` |
| 60 - 100 | `normal` |
| > 100 | `tachycardia` |

### Blood Pressure (AHA/ACC 2017)
| SBP (mmHg) | DBP (mmHg) | Label |
|------------|------------|-------|
| > 180 | OR > 120 | `hypertensive_crisis` |
| >= 140 | OR >= 90 | `hypertension_stage2` |
| >= 130 | OR >= 80 | `hypertension_stage1` |
| 120 - 129 | AND < 80 | `elevated` |
| < 120 | AND < 80 | `normal` |

### Signal Quality (Skewness)
| Condition | Label |
|-----------|-------|
| 0.5 <= skew <= 2.0 | `good_quality` |
| skew < 0.5 or > 2.0 | `noisy_or_distorted` |
| skew ~ 0 | `symmetric_unusual` |

### Sleep Apnea (AHI)
| Value (events/hr) | Label |
|-------------------|-------|
| < 5 | `normal_ahi<5` |
| 5 - 14 | `mild_5<=ahi<15` |
| 15 - 29 | `moderate_15<=ahi<30` |
| >= 30 | `severe_ahi>=30` |

### HRV Metrics

| Metric | Low | Normal | High |
|--------|-----|--------|------|
| SDNN (ms) | < 50 | 50 - 100 | > 100 |
| RMSSD (ms) | < 20 | 20 - 50 | > 50 |
| pNN50 (%) | < 3 | 3 - 25 | > 25 |

### SpO2
| Value (%) | Label |
|-----------|-------|
| >= 95 | `normal` |
| 90 - 94 | `mild_hypoxemia` |
| 85 - 89 | `moderate_hypoxemia` |
| < 85 | `severe_hypoxemia` |

### Respiratory Rate
| Value (breaths/min) | Label |
|---------------------|-------|
| < 12 | `bradypnea` |
| 12 - 20 | `normal` |
| > 20 | `tachypnea` |

---

## File Format

### .mat Structure

```python
{
    'signal': np.ndarray,  # (n_samples, 1250), float32
    'text': np.ndarray,    # (n_samples,), str
    'qa': np.ndarray       # (n_samples,), JSON str
}
```

### QA JSON Format

```json
{
    "heart_rate_category": {
        "question": "What is the heart rate category for this PPG segment?",
        "answer": "normal"
    },
    "blood_pressure_category": {
        "question": "Classify the blood pressure level.",
        "answer": "hypertension_stage1"
    }
}
```

---

## Usage

```python
import scipy.io as sio
import json
import numpy as np

# Load dataset
mat = sio.loadmat('data/ppg_text_qa_v1/dalia_ppg_text.mat')

# Access signal
signal = mat['signal'][0]  # Shape: (1250,), Range: [0, 1]

# Parse QA
qa_str = mat['qa'][0]
if isinstance(qa_str, np.ndarray):
    qa_str = str(qa_str.item())
qa = json.loads(qa_str)

for category, pair in qa.items():
    print(f"{category}: {pair['answer']}")
```

---
