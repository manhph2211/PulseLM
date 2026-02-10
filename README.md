PulseLM
=====

<div align="center" style="font-size: 5em;">
  <strong>PulseLM: A Foundation Dataset and Benchmark for PPG-Text Learning</strong>
  <br> </br> 
</div>

<div align="center"> 
<a href="https://github.com/manhph2211/PULSE-LM/"><img src="https://img.shields.io/badge/Website-QHEART WebPage-blue?style=for-the-badge"></a>
<a href="https://arxiv.org/pdf/2505.XXXX"><img src="https://img.shields.io/badge/arxiv-Paper-red?style=for-the-badge"></a>
<a href="https://huggingface.co/Manhph2211/"><img src="https://img.shields.io/badge/Checkpoint-%F0%9F%A4%97%20Hugging%20Face-White?style=for-the-badge"></a>
</div>

<div align="center">
  <a href="https://github.com/manhph2211/" target="_blank">Hung&nbsp;Manh&nbsp;Pham*</*a> &emsp;
    <a href="" target="_blank">Hung&nbsp;Manh&nbsp;Pham*</*a> &emsp;

  <a href="https://aqibsaeed.github.io/" target="_blank">Aaqib&nbsp;Saeed</a> &emsp;
  <a href="https://www.dongma.info/" target="_blank">Dong&nbsp;Ma</a> &emsp;
</div>
<br>

<div align="center" >
  <small>* equal contribution</small>
  <small>+ equal corresponding authorship</small>
</div>



## Introduction

**PulseLM** is a multimodal framework that integrates PPG (Photoplethysmography) signal encoders with large language models for physiological signal understanding. The project includes a large-scale standardized PPG dataset and a model architecture that combines pretrained PPG encoders with LLM backbones (LLaMA, Qwen variants) via LoRA fine-tuning.

Each sample consists of:
- A **PPG signal** (10-second, 125 Hz, cleaned, processed and normalized)
- A **text description** of the data (metadata, labels, ground information, recording conditions, sensor details, activities, etc.)
- **Question-answer pairs** across 12 clinical-related categories/tasks 

This dataset enables research and applications in:
- PPG Signal Foundation models
- Multimodal PPG-language alignment
- Zero-shot physiological signal classification
- Explainable AI for health monitoring

| Summary | Values |
|--------|-------|
| Total Used Datasets | 15 |
| Total PPG-Text Samples | ~1.31M |
| Total QA Pairs | ~1.31M |
| Signal Length | 10 seconds |
| Sampling Rate | 125 Hz |
| Signal Range | Normalized [0, 1] |
| QA Tasks | 12 |




## Text Description Examples

<!-- Each sample includes a natural language text description in the `text` key that summarizes clinical metadata, physiological measurements, recording context, and signal characteristics, etc. For example: -->
</details>

<details>
<summary><b>Examples</b></summary>


```
A 44-year-old male patient. BMI 28.2. wearable smartwatch PPG recording. normal sinus rhythm, no atrial fibrillation.
```


```
A 34-year-old male. height 182cm, weight 78kg (BMI: 23.5). medium skin, exercises 6 hours/week. Currently unknown activity. Heart rate: 50 bpm.
```

```
In-ear PPG signal recorded from the left ear using green LED. Participant is a 24-year-old female with very light (Type I) skin tone. Recording was made during chewing activity (chewing). Heart rate is 78 bpm (normal).
```


```
Patient diagnosed with normal (AHI < 5). Current segment shows normal breathing. Breathing pattern is regular and unobstructed.
```



```
Blood pressure: 108/80 mmHg (Normal). Heart rate: 110 bpm (tachycardia). Cardiac cycle: 546ms (systolic: 168ms, diastolic: 378ms). Signal quality: acceptable quality. Time to steepest upstroke: 82.3ms. Systolic AUC: 7.3133. Peak-to-peak interval: 68ms.
```


```
This PPG recording was collected while the subject was performing office work activities. Heart rate is 84 bpm (normal). Mean RR interval is 714 ms. RMSSD is 24.3 ms indicating moderate parasympathetic activity. SDNN is 31.3 ms showing reduced heart rate variability. pNN50 is 7.7%.
```


```
A 77.0-year-old m patient. height 160.2cm, weight 67.5kg (BMI: 26.3). from General surgery department. undergoing Colorectal. ASA physical status 2. Blood pressure: 134/58 mmHg (hypertension stage1). mean arterial pressure: 91 mmHg. heart rate: 85 bpm (normal). HRV metrics: MeanNN=704.7ms, SDNN=9.3ms, RMSSD=11.3ms, pNN50=0.0%. medical history: hypertension.
```

```
A 28-year-old male. height 178cm, weight 76kg (BMI: 24.0). Current emotional state: baseline/neutral.
```

```
PPG signal recorded from wrist position. Heart rate is 85 bpm (normal). Mean RR interval is 708 ms. RMSSD is 11.3 ms indicating reduced parasympathetic activity. SDNN is 17.9 ms showing reduced heart rate variability. pNN50 is 0.0%.
```
</details>

## Question Answering Examples

<details>
<summary><b>Examples</b></summary>

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


## Usage

```bash
conda create -n ppg python=3.10 -y
conda activate ppg
pip install -r requirements.txt
```

```python
import scipy.io as sio
import json
import numpy as np

mat = sio.loadmat('path to .mat file')
signal = mat['signal'][0] 
qa_str = mat['qa'][0]
if isinstance(qa_str, np.ndarray):
    qa_str = str(qa_str.item())
qa = json.loads(qa_str)
for category, pair in qa.items():
    print(f"{category}: {pair['answer']}")
```


## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{pham2026pulselm,
  title={PULSE-LM: A Large-Scale PPG Language Model for Multimodal Health Analysis},
  author={},
  journal={arXiv preprint arXiv},
  year={2026},
  }
```
