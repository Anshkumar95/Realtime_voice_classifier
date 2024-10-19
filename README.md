# Realtime_voice_classifier
This model utilizes real speech data for sentiment analysis without converting it to text.

**Approach 1 **
1.	Extract MFCCs from audio.

2.	Preprocess and pad MFCC features.

3.	Design a Transformer-based architecture for processing MFCC features.

4.	Train the model on a labelled sentiment dataset.

5.	Implement real-time processing for live sentiment prediction.

Approach 1 has been implemented in Audio_classifier. 


**Approach 2 **
Data Loading:
	•	We iterated through a dataset of .wav audio files, loading each audio file using librosa and resampling it to 16kHz (as required for the HuBERT model).
	•	The raw audio data and filenames were stored in a DataFrame for easy handling.
Audio Preprocessing:
	•	We used the Wav2Vec2Processor from the Hugging Face library to preprocess the audio data. The processor tokenized the audio signals and converted them into input 		features suitable for the HuBERT model.
Custom Dataset and DataLoader:
	•	We implemented a custom PyTorch Dataset class to handle the input audio and labels.
	•	We used a DataLoader with a custom collate_fn to pad audio sequences to the same length for batch processing, preparing them for model training.
Model Fine-Tuning:
	•	We loaded a pretrained HuBERT model (HubertForSequenceClassification) and fine-tuned it on our dataset. The model’s task was to classify speech into three sentiment classes (assuming 3 labels).
	•	During training, we performed forward passes, calculated the loss, and used backpropagation to update the model’s weights using the AdamW optimizer.
Approach 2 has been implemented in Hubert. 
