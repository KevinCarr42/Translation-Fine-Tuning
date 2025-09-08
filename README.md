# From Data to Translation: Leveraging AI for Efficient and Accurate Translation of Scientific Reports 

# Phase 2: AI Translation Fine-Tuning

## Description

This phase focuses on fine-tuning AI translation models using the cleaned CSAS translation data from Phase 1. During initial fine-tuning attempts, several data quality issues emerged that required additional cleaning steps, including missing punctuation, incomplete sentences, OCR errors, and non-words. These issues were systematically identified and addressed to improve training data quality.

The fine-tuning process involved selecting appropriate base models, experimenting with different hyperparameters, and iteratively refining the training approach based on translation quality results. Multiple model configurations were tested to identify the most promising candidates for scientific translation tasks.

## Key Components

### Data Cleaning and Preparation
- An **[advanced data cleaning pipeline](https://github.com/KevinCarr42/Translation-Fine-Tuning/blob/master/training_data_creation_and_cleaning.ipynb)** was used to handle issues discovered during initial fine-tuning, including:
  - Missing punctuation detection and filtering
  - Incomplete sentence identification
  - OCR error correction and filtering
  - Non-word removal
  - Data quality validation

### Model Fine-Tuning
- Hyperparameter optimisation for translation quality
- Multiple model architecture testing
- Training performance evaluation
- Model selection based on scientific translation accuracy

## Challenges Addressed

1. **Data Quality Issues**: Initial fine-tuning revealed numerous data quality problems not apparent during Phase 1, requiring sophisticated filtering and cleaning approaches.

2. **Scientific Context Preservation**: Ensuring fine-tuned models maintain scientific accuracy while improving translation fluency.

3. **Hyperparameter Optimisation**: Balancing translation quality with training efficiency across different model configurations.

## Outcomes

This phase produced several fine-tuned translation models optimised for CSAS scientific documents, with significantly improved data quality compared to the initial Phase 1 dataset. The models demonstrated enhanced performance on scientific terminology and context-specific translations.

## All Phases

- **Phase 1**: [Data Gathering and Transformation](https://github.com/KevinCarr42/AI-Translation) (complete)
- **Phase 2**: AI Translation Fine-Tuning (complete)
- **Phase 3**: [Rule-Based Preferential Translations](https://github.com/KevinCarr42/rule-based-translation) (complete)
- **Phase 4**: [AI Translation Quality Survey App](https://github.com/KevinCarr42/translation-quality-survey-app) (complete)
- **Phase 5**: [Final AI Translation Model and Translation Quality Evaluation](https://github.com/KevinCarr42/CSAS-Translations) (in-progress)
- **Phase 6**: Deploy the Final Model (in-progress)
