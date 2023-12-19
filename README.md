README - Sleep State Detection using LSTM and U-Net with Attention --

This project involves developing a deep learning model for detecting sleep and wake states using accelerometer data. The core of the model is a combination of LSTM (Long Short-Term Memory) and U-Net architectures, enhanced with attention gating 
mechanisms for improved time-series analysis.

Competition link: https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states


Files and Directories--
- {series_id}_test_series.parquet
    - All data files divided by series_id to deal with mismatched sequence lengths when batching later on.
- train_events.csv: CSV file containing training event data.
- train_series.parquet, test_series.parquet: Parquet files with training and testing series data. May need to download train_series.parquet from kaggle as it is large but if all {series_id}_test_series.parquet files are present then this is not needed.



Components--
- Import dependencies
- Data Preprocessing
    - Credit to this kaggle post for providing some of the preprocessing code: https://www.kaggle.com/code/danielphalen/cmss-grunet-train
    - Any use of the above kaggle code is clearly denoted in the final project notebook.
    - SleepDatasetTrain Class: Handles the loading and preprocessing of time-series data from Parquet files. It applies Gaussian smoothing to the event labels and segments the data into consistent lengths.
    - Data Filtering: Identifies and filters out series with non-continuous data or missing events.
    - Data Splitting: Splits the series IDs into training and testing sets, with an option for demo mode for expedited runs.
- Layer Definitions
    - BiResidualLSTM
    - AttentionGate (deprecated)
    - AttentionGate2
        - A corrected version of the first AttentionGate layer
- Training sections
    - Multiple models to choose from, but final model has section header denoted as 'FINAL MODEL: LSTMAttn2UNet' in notebook.
- Evaluation

Model Architecture--
- FOUND IN THE 'FINAL MODEL' section of our notebook.
- ResidualBiLSTM: A custom LSTM module with residual connections and bidirectional processing capability.
- Attention Mechanisms: Implemented to focus the model on relevant sequences, crucial for detecting sparse events in the data.
- LSTMNET and LSTMAttnUNET: Core model classes that integrate LSTM with U-Net architecture, employing attention mechanisms for enhanced sequence modeling.


Training and Evaluation--
- Data Loaders: Torch data loaders for batching and shuffling the training and testing datasets.
- Model Training: The training process involves feeding the preprocessed data through the LSTM and U-Net models.


To use this code--
1) Ensure all necessary data files are in the same directory. This includes:
    - All .parquet files by SERIES_ID
    - train_events.csv
    - train_series.parquet (not needed if all {series_id} parquet files are already separated) as in first bullet point)
    - test_series.parquet
    - These can all be unzipped from Data/data.zip
2) Change path names to paths where data is located in the PATHS class in the preprocessing section. MAIN_DIR and SPLIT_DIR are the same if all above files are in the same directory.
3) Run the notebook up until just after parameter definition. That is, the first cell after the "Training" header.
4) Go through the notebook and pick a model to train. The final model is labeled as 'FINAL MODEL' in the section header.
5) Run the scoring code in the 'Evaluation' section.


Dependencies--
- torch
- pandas
- numpy
- matplotlib
- pyarrow
- tqdm
- scikit-learn
- math
- random
- gc
- ctypes
- copy


Custom Dependence--
