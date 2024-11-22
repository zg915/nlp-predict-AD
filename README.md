# NLP Predict AD: Project Workflow

## Step 1: Retrieve Downloaded Dataset (`raw_data_transform`)
- **Input**: Files downloaded from `Talkbank - Pitt/`
- **Output**: JSON file of Pitt Cookie Theft Task raw data 
  - `data/pitt_cookie_dataset.json`
- **Description**: Transform raw files into a standardized JSON format.

---

## Step 2: Data Cleaning (`data_cleaning`)
- **Input**: `data/pitt_cookie_dataset.json`
- **Outputs**: 
  - `data/train_v2_400.csv`
  - `data/test_v2_149.csv`
- **Process**:
  - Remove special tokens and error codes.
  - Select only utterance data.
  - Perform train-test split (400 vs. 149 samples).
     - Ensure train set is balanced (200 vs. 200).
  - Versions:
    - **v1**: Cleaned utterances containing only patient speech.
    - **v2**: Cleaned utterances including both investigator and patient speech.

---

## Step 3: Data Generation (`data_generation`)
- **Input**: `data/train_v2_400.csv`
- **Output**: Generated data files:
  - `data/xxx/transcription_[model_type]_[batch_index].json`
- **Process**:
  - Use GPT models to generate new datapoints.
    - Models: `gpt-4o-2024-08-06`, `chatgpt-4o-latest`.
  - Prompts:
    - **v1**: Zero-shots. No example in the prompt.
    - **v2**: Examples only contain patient transcripts. Contains instructions about using (.) (..) to represent pauses.
    - **v3**: Deletes some instructions for special tokens usage.
    - **v4**: A comparable version for patient and control data.

---

## Step 4: Data Processing (`data_processing`)
- **Inputs**: 
  - `data/xxx/transcription_[model_type]_[batch_index].json`
  - `data/train_v2_400.csv`
  - `data/test_v2_149.csv`
- **Outputs**: 
  - `data/train_complete_v1_800.csv`
  - `data/test_complete_v1_149.csv`
- **Process**:
  - Concatenate generated data with the true training set.
  - Remove speech from investigators.
  - Remove any remaining special tokens.

---

## Step 5: Similarity Calculation (`similarity_calculation`)
- **Input**: `data/train_complete_v1_800.csv`
- **Output**: Similarity metrics
- **Process**:
  - Calculate similarity between data points using:
    - Cosine similarity.
    - Unique word comparison.

---

## Step 6: BERT Modeling (`bert_finetune`)
- **Inputs**:
  - `data/train_complete_v1_800.csv`
  - `data/test_complete_v1_149.csv`
- **Output**: Fine-tuned BERT model.
- **Process**:
  - Use various combinations of training datasets to fine-tune BERT for classification.
  - Compare results across datasets.
