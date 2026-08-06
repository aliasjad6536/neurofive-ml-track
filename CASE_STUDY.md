# Case Study: Predicting Employee Attrition

## The Problem

Losing a skilled employee is expensive — replacing someone typically costs 6 to 9 months of their salary once recruiting, onboarding, and lost productivity are factored in. Most companies only find out someone is leaving when they hand in their resignation, which is too late to do anything but manage the exit. In this dataset of 1,470 employees, roughly 1 in 6 (16%) had left the company, and the data shows their departures weren't random — they followed clear, learnable patterns.

## The Approach

I built a machine learning pipeline that predicts attrition risk from data most HR systems already collect: job role, department, compensation, tenure, satisfaction scores, and whether someone regularly works overtime. Two engineered features sharpened the signal further — `income_per_year_worked` (whether pay has kept pace with someone's overall experience) and `promotion_stagnation` (how long someone has gone without upward movement, relative to their tenure). After comparing a Logistic Regression model against a Random Forest, the Logistic Regression model performed better at actually catching at-risk employees (62% recall on employees who left, versus 32% for Random Forest), so it was chosen as the final deployed model, reaching 85.7% overall accuracy on employees the model had never seen during training.

## The Business Value

Instead of losing employees silently, an HR or people-ops team could run this model quarterly (or whenever employee data updates) and get back a ranked list of employees showing elevated attrition risk. The clearest patterns in the data were revealing: employees who work overtime, earn less relative to their experience, and have gone the longest without a promotion are the most likely to leave. That turns attrition from a reactive HR problem into a proactive one — a manager could have a career conversation, review someone's workload, or fast-track a promotion review for a flagged employee weeks or months before they'd otherwise submit their resignation. Even catching a modest fraction of departures early, at 6-9 months of salary saved per retained employee, would easily justify running a model like this as a standard part of an HR analytics toolkit.

## Limitations

This model isn't a replacement for actual management judgment — it's a triage tool. It will miss some departures and flag some employees who were never going to leave (the confusion matrix in the notebook shows both). It also reflects patterns specific to this dataset's company and time period; retraining periodically on a company's own current data would be necessary before relying on it operationally.
