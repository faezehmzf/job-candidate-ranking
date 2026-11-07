# Data

The CSV files are not included in this repository. Put them in this folder:

| file | rows | content |
|---|---|---|
| `jobs.csv` | 5,000 | job postings: title, required skills, experience range, salary range + currency, location, remote policy, posting date, company size, industry |
| `candidates.csv` | 50,000 | candidate profiles: experience, current title, skills, education, certifications, English level, expected salary, location, relocation, profile completeness, account date |
| `applications_train.csv` | 118,772 | `application_id, job_id, candidate_id, application_date, relevance_label` (0-4), dates 2023-01-01 .. 2024-06-30 |
| `applications_test.csv` | 52,700 | same without the label, dates 2024-07-01 .. 2025-01-30 |
| `applications_test_labels.csv` | 52,700 | test applications with `relevance_label`; used only for the final evaluation in `scripts/04_select_and_test.py` |

All records are synthetic. Skills and multi-valued fields are pipe-separated (`Python|SQL|Docker`).
