# NimbusPay-Ticket-Triage
Fine-tuning a small language model for support ticket triage

## Project Overview

This project focuses on fine-tuning a small language model to classify and prioritize customer support tickets for a fictional payments application, NimbusPay.

The objective is to convert unstructured support messages into structured JSON records containing the ticket category, priority, transaction details, language, and whether human intervention is required.

The project covers training data auditing, label correction, baseline evaluation, QLoRA fine-tuning, development-set evaluation, and test-set prediction.

## Model and Tools

- **Base model:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** QLoRA
- **Quantization:** 4-bit NF4 with double quantization
- **Adapter:** LoRA
- **Framework:** Hugging Face Transformers and PEFT
- **Environment:** Google Colab with NVIDIA T4 GPU
- **Language:** Python

## Dataset Preparation

A total of 1,445 training records were reviewed against the provided NimbusPay labelling specification.

The audit identified and addressed:
- 25 malformed JSON labels
- 70 incorrect or missing field names
- 65 priority-label inconsistencies
- 30 date-format inconsistencies
- 20 transaction ID formatting issues

A total of 210 cleaning actions were recorded in `cleaning_log.csv`.

The cleaned dataset passed the final structural and semantic validation checks.

## Fine-Tuning Configuration

The model was fine-tuned using 4-bit QLoRA.

| Parameter | Configuration |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Learning rate | 0.0002 |
| Epochs | 2 |
| Maximum sequence length | 2,304 |
| Trainable parameters | 9,232,384 |
| Training time | 32.34 minutes |
| Peak GPU memory | 6.18 GB |

Assistant-only loss masking was used during training so that loss was calculated on assistant responses rather than the system and user messages.

## Evaluation Results

The fine-tuned model was evaluated on 200 development examples.

| Metric | Result |
|---|---:|
| Valid JSON | 100.0% |
| Exact match accuracy | 82.5% |
| Mean field accuracy | 96.9% |

The model performed well on structured field extraction, particularly transaction IDs and language identification.

Priority classification and human-escalation decisions accounted for most of the remaining errors.

## Error Analysis

The main error patterns identified were:

- Incorrect priority assignment around business-rule thresholds.
- Inconsistent human-escalation decisions.
- Occasional conversion errors between rupees and paise.
- Missed fraud-category overrides in a small number of cases.

An error analysis of 35 incorrect development predictions was documented separately.

## Experiments

The completed experiment is the cleaned-data 4-bit QLoRA baseline.

Additional experiment configurations were prepared to investigate raw versus cleaned data, FP16 LoRA versus 4-bit QLoRA, and alternative LoRA settings. These experiments were not executed because of Google Colab GPU usage restrictions.

No results have been assumed or fabricated for the unfinished experiments.

## Test Predictions

The fine-tuned model was used to generate predictions for all 400 hidden-test records.

The predictions are provided in `predictions_test.jsonl`. The generated outputs were retained without corrective post-processing.

## Repository Structure

```text
NimbusPay-Ticket-Triage/
│
├── NimbusPay_Ticket_Triage_organized.ipynb
├── cleaning_log.csv
├── error_analysis.csv
├── experiment_results.csv
├── predictions_test.jsonl
└── README.md
```

## Limitations and Future Improvements

Further work could focus on:

1. Improving priority classification using more examples around business-rule thresholds.
2. Improving fraud detection and escalation decisions.
3. Evaluating different LoRA configurations and training settings.
4. Completing the remaining controlled experiments.
5. Recovering or recreating the trained adapter for reproducibility.

## Acknowledgement

This project was completed as part of the Machine Learning Intern take-home assessment for VIMA3YA.
