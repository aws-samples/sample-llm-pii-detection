# Benchmark reference

Reference material for the blog post *Model-agnostic PII detection with LLMs*.
This is the detail that the post links out to: the full detector benchmark
table, the per-dataset label mappings, and the Extended-configuration category
definitions.

- The **detection system prompt** and the standard **PII / do-not-flag category
  definitions** are the shipped source of truth in
  [`pii_detector/templates.py`](../pii_detector/templates.py)
  (`PII_DETECTION_SYSTEM_PROMPT`, `PII_DEFINITIONS`, `PUBLIC_CATEGORIES_DICT`).
  They are not duplicated here so they cannot drift from the code.

## Datasets

Five public PII corpora from Hugging Face, each carrying ground-truth spans;
roughly 10,000 rows sampled per dataset. Together: 49,365 records and 222,114
ground-truth core spans across eight languages (de, en, es, fr, hi, it, nl, te).

| # | Dataset | Records | Core GT | Notes |
|---|---------|--------:|--------:|-------|
| 1 | [ai4privacy_500k](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) | 9,947 | 23,822 | Multilingual; adds sex/gender, organisation |
| 2 | [ai4privacy](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) | 9,936 | 70,720 | 6 langs; names / addresses / email / phone |
| 3 | [gretel](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1) | 9,991 | 41,967 | English; HR / financial / customer-service docs |
| 4 | [isotonic](https://huggingface.co/datasets/Isotonic/pii-masking-200k) | 9,498 | 21,674 | 15+ extra domain categories |
| 5 | [nemotron](https://huggingface.co/datasets/nvidia/Nemotron-PII) | 9,993 | 63,931 | US/UK; 30+ raw entity categories |
|   | **Total** | **49,365** | **222,114** | |

A predicted span is matched to ground truth by exact `(start, end, label)`
overlap (IoU = 1.0) and scored with Precision, Recall, and F1.

## Canonical "core" taxonomy

Every raw label (detector output and dataset ground truth alike) is mapped onto
one canonical taxonomy, and each detector is scored only on the intersection of
the label scopes it and the dataset both declare.

| Canonical entity | Covers |
|------------------|--------|
| NAME | Private and public person names |
| ADDRESS | Full and partial addresses, locations |
| CONTACT_INFO | Email and phone |
| DATE | Birth dates, appointments, anniversaries |
| AGE | Age with unit |
| SSN | Social-security and national numbers |
| FINANCIAL | Credit card and bank account |
| IP_ADDRESS | IPv4, IPv6, MAC |
| URL | Public and private URLs |
| USERNAME | Usernames |
| PASSWORD | Passwords, PINs, access keys |
| ID_NUMBER | Passports, licences, customer/employee IDs |

## All detectors — Core F1 and latency

Every detector benchmarked, with span-level Core F1 (all five datasets, 49,365
records) and estimated per-detection latency. Baseline first; LLMs grouped by
backend and sorted by Core F1. Latency is the total wall-clock time over the run
extrapolated back to a single record (Bedrock at 24 concurrent workers, EC2
single-stream on the listed instance), so it is indicative rather than a strict
single-call measurement. The two smallest open models mark the capability floor
below which the task breaks down.

| # | Detector | Backend | Instance | Core F1 | Per-detection (s) |
|---|----------|---------|----------|--------:|------------------:|
| 1 | PrivacyFilter (baseline) | EC2 | g4dn.xlarge | 80.7% | 2.15 |
| 2 | Claude Sonnet 4.6 | Bedrock | — | 84.2% | 2.29 |
| 3 | Mistral Large 3 | Bedrock | — | 83.1% | 1.16 |
| 4 | Claude Haiku 4.5 | Bedrock | — | 82.4% | 1.34 |
| 5 | OSS-GPT 20B | Bedrock | — | 81.3% | 13.02 |
| 6 | OSS-GPT 120B | Bedrock | — | 79.4% | 3.91 |
| 7 | Nova Lite 2 | Bedrock | — | 74.9% | 0.77 |
| 8 | openai/gpt-oss-20b | EC2 | g5.12xlarge | 81.6% | 1.17 |
| 9 | Qwen3.6-27B | EC2 | g5.12xlarge | 79.5% | 12.79 |
| 10 | Qwen3.6-35B-A3B | EC2 | g5.12xlarge | 79.4% | 5.59 |
| 11 | Gemma-4-E4B-it | EC2 | g5.xlarge | 79.4% | 0.43 |
| 12 | Gemma-4-26B-A4B-it | EC2 | g5.12xlarge | 77.3% | 0.44 |
| 13 | Qwen3.5-9B | EC2 | g5.12xlarge | 76.4% | 15.31 |
| 14 | Qwen3.5-4B | EC2 | g5.xlarge | 76.2% | 10.23 |
| 15 | Gemma-4-E2B-it | EC2 | g5.xlarge | 70.1% | 0.20 |
| 16 | Qwen3.5-2B | EC2 | g5.xlarge | 51.5% | 0.77 |
| 17 | Qwen3.5-0.8B | EC2 | g5.xlarge | 1.2% | 0.13 |

## Extended configuration — base vs. Ext

Base prompt vs. the Extended ("Ext") configuration, on the five public datasets,
sorted by Extended-entity F1. Adding the extra category definitions lifts
extended-entity F1 roughly six-fold and also nudges core F1 up. This works on
all tested models; no fixed-scope baseline can do it at all.

| # | Detector | Backend | Extended-entity F1 (base ▸ Ext) | Core F1 (base ▸ Ext) |
|---|----------|---------|---------------------------------|----------------------|
| 1 | Claude Sonnet 4.6 | Bedrock | 12.7% ▸ 80.8% | 84.2% ▸ 88.1% |
| 2 | Qwen3.6-35B-A3B | EC2 | 9.4% ▸ 80.5% | 79.4% ▸ 83.5% |
| 3 | openai/gpt-oss-20b | EC2 | 12.1% ▸ 73.3% | 81.6% ▸ 83.1% |
| 4 | Mistral Large 3 | Bedrock | 17.3% ▸ 72.7% | 83.1% ▸ 89.1% |
| 5 | Gemma-4-E4B-it | EC2 | 12.5% ▸ 72.5% | 79.4% ▸ 83.8% |

## Per-language and per-entity accuracy (ai4privacy_500k)

Drawn from the ai4privacy_500k breakdown. `openai/gpt-oss-20b` stays in a tight
83–90% Core-F1 band across all eight languages, including non-Latin Hindi and
Telugu where regex/translation baselines degrade, and scores at or above the
frontier models on the highest-stakes identifiers. The shared weak spot across
the LLM detectors is DATE, where span boundaries and formats are genuinely
ambiguous.

| Entity | gpt-oss-20b Core F1 |
|--------|--------------------:|
| SSN | 99.1% |
| FINANCIAL | 97.1% |
| ID_NUMBER | 95.9% |
| CONTACT_INFO | 94.8% |
| NAME | 92.7% |
| DATE | ~50% |

## Per-dataset label mapping

Each dataset's raw ground-truth labels mapped onto the canonical taxonomy.
Categories outside the common core are grouped as **Extended**.

### ai4privacy_500k
| Canonical | Raw ground-truth labels |
|-----------|-------------------------|
| NAME | givenname, surname, title |
| ADDRESS | buildingnum, city, street, zipcode |
| CONTACT_INFO | email, telephonenum |
| DATE | date, time |
| AGE | age |
| SSN | socialnum |
| FINANCIAL | creditcardnumber |
| ID_NUMBER | driverlicensenum, idcardnum, passportnum, taxnum |
| Extended | gender, sex |

### ai4privacy
| Canonical | Raw ground-truth labels |
|-----------|-------------------------|
| NAME | givenname1/2, lastname1/2/3, title |
| ADDRESS | building, city, country, geocoord, postcode, secaddress, state, street |
| CONTACT_INFO | email, tel |
| DATE | bod, date, time |
| SSN | socialnumber |
| FINANCIAL | cardissuer |
| ID_NUMBER | driverlicense, idcard, passport |
| IP_ADDRESS | ip |
| USERNAME | username |
| PASSWORD | pass |
| Extended | sex |

### gretel
| Canonical | Raw ground-truth labels |
|-----------|-------------------------|
| NAME | first_name, last_name, name |
| ADDRESS | address, city, coordinate, country, postcode, state, street_address |
| CONTACT_INFO | email, phone_number |
| DATE | date, date_of_birth, date_time, time |
| SSN | ssn |
| FINANCIAL | account_number, bank_routing_number, credit_card_number, cvv, swift_bic |
| ID_NUMBER | customer_id, employee_id, medical_record_number, national_id, tax_id, license_plate, ... |
| IP_ADDRESS | ipv4, ipv6 |
| URL | url |
| USERNAME | user_name |
| PASSWORD | api_key, password, pin |
| Extended | company_name |

### isotonic
| Canonical | Raw ground-truth labels |
|-----------|-------------------------|
| NAME | firstname, lastname, middlename, prefix |
| ADDRESS | buildingnumber, city, county, nearbygpscoordinate, secondaryaddress, state, street, zipcode |
| CONTACT_INFO | email, phoneimei, phonenumber |
| DATE | date, dob |
| AGE | age |
| SSN | ssn |
| FINANCIAL | accountname, accountnumber, creditcardcvv, creditcardnumber, iban, maskednumber |
| ID_NUMBER | — |
| IP_ADDRESS | ip, ipv4, ipv6, mac |
| URL | url |
| USERNAME | username |
| PASSWORD | password, pin |
| Extended | amount, bitcoinaddress, companyname, ethereumaddress, jobtitle, jobtype, useragent, vehiclevin, sex, ... |

### nemotron
| Canonical | Raw ground-truth labels |
|-----------|-------------------------|
| NAME | first_name, last_name |
| ADDRESS | city, coordinate, country, county, postcode, state, street_address |
| CONTACT_INFO | email, fax_number, phone_number |
| DATE | date, date_of_birth, date_time, time |
| AGE | age |
| SSN | ssn |
| FINANCIAL | account_number, bank_routing_number, credit_debit_card, cvv, swift_bic |
| ID_NUMBER | customer_id, employee_id, medical_record_number, national_id, tax_id, license_plate, ... |
| IP_ADDRESS | ipv4, ipv6, mac_address |
| URL | url |
| USERNAME | user_name |
| PASSWORD | api_key, password, pin |
| Extended | blood_type, company_name, education_level, employment_status, gender, occupation, political_view, race_ethnicity, religious_belief, sexuality, http_cookie |

## Extended-entity category definitions

The extra category definitions injected into the prompt for the Extended
("Ext") configuration, per dataset. Each is added verbatim to the
`{pii_categories}` block (via `extra_definitions`), alongside a few worked
examples and the suppression of the conflicting do-not-flag lines (via
`extra_public_drops`). See the `with_extra_definitions` example in
[`examples/detect.py`](../examples/detect.py).

### nemotron / gretel
| Category | Definition (abridged) |
|----------|------------------------|
| OCCUPATION | Any job title, profession, role, or role label: multi-word titles and bare single-word roles ("data scientist", "Owner", "Technician", "Student"). |
| COMPANY_NAME | Any company, employer, brand, or organization name, flagged on every occurrence (gretel injects this category only). |
| EDUCATION_LEVEL | Educational attainment, degree, or schooling status ("high school diploma", "bachelor's", "PhD"). |
| EMPLOYMENT_STATUS | Employment state or work arrangement ("employed", "full-time", "contractor", "retired"). |
| POLITICAL_VIEW | Political affiliation, ideology, or partisan stance ("Democrat", "libertarian"). |
| RACE_ETHNICITY | Race, ethnicity, or ethnic background ("Black", "Hispanic", "Italian-American"). |
| SEXUALITY | Sexual orientation ("gay", "bisexual", "asexual"). |
| RELIGION / RELIGIOUS_BELIEF | Religious affiliation, or a specific belief/practice ("Catholic", "observes Shabbat"). |
| BLOOD_TYPE | Blood type, ABO group and Rh factor ("O+", "AB negative"). |
| GENDER | Gender identity, including bare descriptors stated for a person ("female", "non-binary", "transgender man"). |
| HTTP_COOKIE | Personal HTTP cookie values that identify or track a user session. |

### ai4privacy_500k
| Category | Definition (abridged) |
|----------|------------------------|
| SEX | The binary biological token only (e.g. "Female", "M"). |
| GENDER | Anything beyond binary sex (identity, transgender status, non-binary). |
| ORGANISATION | An organisation, employer, or institution referenced in connection with a person. |

### isotonic
| Category | Definition (abridged) |
|----------|------------------------|
| SEX / GENDER | As above (biological token versus identity). |
| BITCOIN_ADDRESS | A Bitcoin wallet address (base58/bech32, starts with 1/3/bc1). |
| ETHEREUM_ADDRESS | An Ethereum wallet address (42-char hex, 0x-prefixed). |
| AMOUNT | The numeric portion of a monetary amount, excluding currency symbol ("1250.00", "248k"). |
| CURRENCY_SYMBOL | An explicit currency symbol or ISO code ("$", "€", "USD", "BTC"). |
| JOB_TITLE | A specific job title or role descriptor ("Senior Software Engineer"). |
| JOB_TYPE | A class of work arrangement ("full-time", "contract", "freelance"). |
| COMPANY_NAME | Company, employer, or brand name (shared with the nemotron definition). |
| VEHICLE_IDENTIFIER | A vehicle-tied identifier for a person: VIN, plate/registration, or fleet ID. |
| USER_AGENT | A browser/client User-Agent string identifying a user's device or software. |

---

*Sources: `PII_benchmark_10k.xlsx` (5 datasets, 49,365 core records, span-level
F1; accuracy, latency, and Extended-configuration figures) and
`PII_bench_param_1k_restrict.xlsx` (1k rows/dataset). Datasets: ai4privacy_500k,
ai4privacy, gretel, isotonic, nemotron (Hugging Face). Metric: exact-match
span-level Precision/Recall/F1.*
