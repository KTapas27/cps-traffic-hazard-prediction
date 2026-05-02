\# CPS: AI-Driven Traffic \& Climate Hazard Prediction System



\*\*Author:\*\* Tapas Kammula  

\*\*Institution:\*\* BITS Pilani Dubai Campus  

\*\*Target Venue:\*\* IEEE ITSC 2026  



\## Overview

A machine learning system that predicts traffic accident severity with a focus on UAE-specific climate hazards, particularly sandstorms. The system simulates how sandstorm conditions (low visibility, high wind speed) affect accident severity using a model trained on 1.5 million accident records.



\## Key Results

| Model | Accuracy | Macro F1 |

|-------|----------|----------|

| Logistic Regression | 34% | 0.22 |

| Random Forest | 85% | 0.40 |

| XGBoost (Best) | 72% | 0.43 |



\*\*After SMOTE balancing (XGBoost):\*\*

| Severity | Recall Before | Recall After |

|----------|--------------|--------------|

| 1 (Minor) | 0.05 | 0.76 |

| 2 (Moderate) | 0.97 | 0.77 |

| 3 (Serious) | 0.34 | 0.51 |

| 4 (Fatal) | 0.05 | 0.41 |



\## UAE Sandstorm Simulation

Applying UAE sandstorm conditions (visibility < 0.3mi, wind > 35mph):

\- Fatal accidents (Severity 4) increased by 7.5%

\- Serious accidents (Severity 3) increased by 6.7%

\- Overall severity increase: 1.4%



\## Top Features (SHAP Analysis)

1\. Distance(mi) — accident impact spread

2\. Month — seasonal patterns

3\. Wind\_Speed(mph) — sandstorm proxy

4\. Visibility(mi) — sandstorm proxy

5\. Weather\_Condition — environmental factor



\## Project Structure

