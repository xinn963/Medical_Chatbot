"""
Medical Chatbot Web Interface
Flask backend API for serving TensorFlow and PyTorch medical chatbot models
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import sys
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define BERT model class for PyTorch
class BertMedicalClassifier(nn.Module):
    def __init__(self, num_classes=41, model_name='bert-base-uncased'):
        super(BertMedicalClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MedicalChatbot:
    def __init__(self):
        self.tf_model = None
        self.tf_tokenizer = None
        self.tf_label_encoder = None
        self.pytorch_model = None
        self.pytorch_tokenizer = None
        self.pytorch_label_encoder = None
        
        # Define greeting patterns and responses
        self.greeting_patterns = {
            'hello': ['hello', 'hi', 'hey', 'hiya', 'greetings'],
            'how_are_you': ['how are you', 'how do you do', 'how are you doing', 'whats up', "what's up"],
            'good_morning': ['good morning', 'morning'],
            'good_evening': ['good evening', 'evening'],
            'good_night': ['good night', 'goodnight', 'night'],
            'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'take care'],
            'thank_you': ['thank you', 'thanks', 'thank u', 'thx', 'appreciate it'],
            'introduction': ['who are you', 'what are you', 'introduce yourself', 'tell me about yourself']
        }
        
        self.greeting_responses = {
            'hello': [
                "Hello! I'm your medical assistant. How can I help you with your health questions today?",
                "Hi there! I'm here to help you with medical information. What would you like to know?",
                "Hello! I'm a medical chatbot trained to assist with health-related questions. How may I assist you?"
            ],
            'how_are_you': [
                "I'm doing well, thank you for asking! I'm here and ready to help you with any medical questions you have.",
                "I'm functioning perfectly and ready to assist you with health-related information. How can I help you today?",
                "Thank you for asking! I'm here to provide medical assistance. What health topic can I help you with?"
            ],
            'good_morning': [
                "Good morning! I hope you're having a healthy start to your day. How can I assist you with medical information?",
                "Good morning! I'm here to help with any health questions you might have today."
            ],
            'good_evening': [
                "Good evening! I'm here to help you with any medical questions you have. How can I assist you?",
                "Good evening! How can I help you with health-related information tonight?"
            ],
            'good_night': [
                "Good night! Sleep well and take care of your health. I'll be here if you need medical assistance.",
                "Good night! Remember to get adequate rest for good health. Sweet dreams!"
            ],
            'goodbye': [
                "Goodbye! Take care of your health, and don't hesitate to ask if you have more medical questions.",
                "See you later! Remember to consult healthcare professionals for serious medical concerns. Stay healthy!",
                "Farewell! I hope I was helpful with your medical questions. Take care!"
            ],
            'thank_you': [
                "You're very welcome! I'm glad I could help with your medical questions.",
                "Happy to help! If you have more health-related questions, feel free to ask.",
                "You're welcome! Remember to always consult healthcare professionals for serious medical concerns."
            ],
            'introduction': [
                "I'm a medical chatbot designed to provide general health information and answer medical questions. I'm trained on medical data to help you understand symptoms, treatments, and health conditions. However, I'm not a replacement for professional medical advice.",
                "I'm your AI medical assistant! I can help answer questions about health conditions, symptoms, treatments, and general medical information. While I'm knowledgeable about medical topics, always consult with healthcare professionals for personalized medical advice.",
                "Hello! I'm a medical AI assistant trained to provide health information. I can help explain medical conditions, symptoms, and general health topics, but I always recommend consulting with qualified healthcare providers for medical diagnosis and treatment."
            ]
        }
        self.is_tf_loaded = False
        self.is_pytorch_loaded = False
        self.load_tensorflow_model()
        self.load_pytorch_model()
        
    def load_tensorflow_model(self):
        """Load TensorFlow model, tokenizer, and label encoder"""
        try:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tensorflow_version')
            
            # Load tokenizer
            tokenizer_path = os.path.join(base_path, 'medical_tokenizer.pkl')
            with open(tokenizer_path, 'rb') as f:
                self.tf_tokenizer = pickle.load(f)
            logger.info("✅ TensorFlow tokenizer loaded successfully")
            
            # Load label encoder
            label_encoder_path = os.path.join(base_path, 'medical_label_encoder.pkl')
            with open(label_encoder_path, 'rb') as f:
                self.tf_label_encoder = pickle.load(f)
            logger.info("✅ TensorFlow label encoder loaded successfully")
            
            # Create model architecture and load weights
            num_labels = len(self.tf_label_encoder.classes_)
            vocab_size = 30000  # Same as notebook
            max_length = 128    # Same as notebook
            embedding_dim = 128 # Same as notebook
            
            # Recreate model architecture (same as notebook)
            inputs = tf.keras.layers.Input(shape=(max_length,), name='input_sequences')
            
            embedding = tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_length,
                mask_zero=True,
                name='word_embedding'
            )(inputs)
            
            lstm1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                name='bidirectional_lstm_1'
            )(embedding)
            
            lstm2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                name='bidirectional_lstm_2'
            )(lstm1)
            
            dense1 = tf.keras.layers.Dense(256, activation='relu', name='dense_1')(lstm2)
            dropout1 = tf.keras.layers.Dropout(0.3, name='dropout_1')(dense1)
            
            dense2 = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(dropout1)
            dropout2 = tf.keras.layers.Dropout(0.2, name='dropout_2')(dense2)
            
            outputs = tf.keras.layers.Dense(
                num_labels, 
                activation='softmax', 
                name='medical_classification'
            )(dropout2)
            
            self.tf_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='medical_classifier')
            
            # Compile the model
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            self.tf_model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load the trained weights
            weights_path = os.path.join(base_path, 'medical_models', 'best_medical_model.weights.h5')
            self.tf_model.load_weights(weights_path)
            logger.info("✅ TensorFlow model architecture created and weights loaded successfully")
            
            self.is_tf_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Error loading TensorFlow model: {str(e)}")
            self.is_tf_loaded = False
    
    def load_pytorch_model(self):
        """Load PyTorch BERT model and tokenizer"""
        try:
            base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pytorch_version')
            
            # Load tokenizer (using BERT tokenizer)
            self.pytorch_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            logger.info("✅ PyTorch BERT tokenizer loaded successfully")
            
            # Load label encoder
            label_encoder_path = os.path.join(base_path, 'medical_label_encoder.pkl')
            with open(label_encoder_path, 'rb') as f:
                self.pytorch_label_encoder = pickle.load(f)
            logger.info("✅ PyTorch label encoder loaded successfully")
            
            # Load BERT model
            model_path = os.path.join(base_path, 'medical_models', 'bert_best_model.pt')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize model with correct number of classes
            num_classes = len(self.pytorch_label_encoder.classes_)
            self.pytorch_model = BertMedicalClassifier(num_classes=num_classes)
            
            # Load the trained weights
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.pytorch_model.load_state_dict(checkpoint)
            
            self.pytorch_model.to(device)
            self.pytorch_model.eval()
            
            logger.info("✅ PyTorch BERT model loaded successfully")
            self.is_pytorch_loaded = True
            
        except Exception as e:
            logger.error(f"❌ Error loading PyTorch model: {str(e)}")
            self.is_pytorch_loaded = False
    
    def detect_greeting(self, message):
        """Detect if the message is ONLY a greeting (not a medical question with greeting)"""
        message_lower = message.lower().strip()
        
        # If message is very long (>20 words), it's likely not just a greeting
        words = message_lower.split()
        if len(words) > 20:
            return None
        
        # Check for medical keywords that indicate this is a medical question, not just greeting
        medical_keywords = [
            'pain', 'hurt', 'ache', 'sick', 'disease', 'symptom', 'medicine', 'drug', 'medication',
            'antibiotic', 'treatment', 'doctor', 'hospital', 'clinic', 'diagnosis', 'infection',
            'fever', 'cough', 'rash', 'headache', 'nausea', 'vomiting', 'diarrhea', 'allergy',
            'blood', 'pressure', 'diabetes', 'cancer', 'surgery', 'prescription', 'dose',
            'side effect', 'reaction', 'swelling', 'inflammation', 'chronic', 'acute'
        ]
        
        # If any medical keywords are present, treat as medical question, not greeting
        for keyword in medical_keywords:
            if keyword in message_lower:
                return None
        
        # Check for question words that suggest medical inquiry
        question_indicators = ['can', 'should', 'would', 'could', 'what', 'how', 'why', 'when', 'where', 'is', 'are', 'do', 'does']
        has_question_word = any(word in words for word in question_indicators)
        
        # If it has question words AND contains medical context, it's not just a greeting
        if has_question_word and len(words) > 5:
            return None
        
        # Now check for pure greeting patterns (only for short messages)
        for greeting_type, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                # For very short messages, check if it's primarily a greeting
                if len(words) <= 5:
                    if pattern in message_lower:
                        # Return a random response from the appropriate category
                        import random
                        responses = self.greeting_responses[greeting_type]
                        return random.choice(responses)
                # For longer messages, only match if greeting is at the very beginning
                elif message_lower.startswith(pattern + ' ') or message_lower.startswith(pattern + ','):
                    # Check if rest of message after greeting is medical
                    remaining_text = message_lower.replace(pattern, '', 1).strip(' ,')
                    if any(keyword in remaining_text for keyword in medical_keywords):
                        return None  # It's a medical question with polite greeting
        
        return None  # Not a greeting

    def detect_emergency(self, message):
        """Detect medical emergencies that require immediate attention"""
        message_lower = message.lower().strip()
        
        emergency_keywords = [
            # Chest/Heart issues
            'chest pain', 'heart attack', 'cardiac arrest', 'severe chest pain',
            # Breathing issues
            'trouble breathing', 'difficulty breathing', 'can\'t breathe', 'cannot breathe',
            'shortness of breath', 'choking', 'suffocating',
            # Neurological emergencies
            'stroke', 'seizure', 'convulsion', 'unconscious', 'unresponsive',
            'severe headache', 'sudden weakness', 'paralysis', 'can\'t move',
            # Severe bleeding/trauma
            'bleeding heavily', 'severe bleeding', 'profuse bleeding', 'major injury',
            'severe trauma', 'broken bone', 'fractured skull',
            # Poisoning/overdose
            'overdose', 'poisoning', 'swallowed poison', 'toxic',
            # Allergic reactions
            'severe allergic reaction', 'anaphylaxis', 'swelling throat',
            # Mental health emergencies
            'suicidal', 'want to die', 'kill myself', 'end my life',
            # General emergency indicators
            'emergency', 'urgent', 'critical', 'life threatening',
            'severe pain', 'excruciating pain', 'unbearable pain'
        ]
        
        # Check for emergency keywords
        for keyword in emergency_keywords:
            if keyword in message_lower:
                return True
        
        # Check for emergency phrases
        emergency_phrases = [
            'call 911', 'call 999', 'need ambulance', 'going to die',
            'losing consciousness', 'severe symptoms', 'getting worse rapidly'
        ]
        
        for phrase in emergency_phrases:
            if phrase in message_lower:
                return True
                
        return False

    def detect_inappropriate_language(self, message):
        """Detect complaints, inappropriate language, or non-medical content"""
        message_lower = message.lower().strip()
        
        # Complaint patterns
        complaint_keywords = [
            'terrible', 'awful', 'horrible', 'worst', 'useless', 'stupid',
            'garbage', 'trash', 'suck', 'sucks', 'hate', 'disgusting',
            'disappointed', 'unsatisfied', 'complaint', 'complain'
        ]
        
        # Service-related complaints
        service_complaints = [
            'your service', 'this service', 'this app', 'this chatbot',
            'you are', 'you\'re terrible', 'you\'re stupid', 'you suck'
        ]
        
        # Inappropriate/profane language (keeping it mild for medical context)
        inappropriate_words = [
            'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron',
            'shut up', 'go away', 'leave me alone'
        ]
        
        # Check for complaint patterns
        for keyword in complaint_keywords:
            if keyword in message_lower:
                for service in service_complaints:
                    if service in message_lower:
                        return 'complaint'
        
        # Check for inappropriate language
        for word in inappropriate_words:
            if word in message_lower and not self._has_medical_context(message_lower):
                return 'inappropriate'
        
        # Check for completely non-medical content
        if self._is_completely_non_medical(message_lower):
            return 'non_medical'
            
        return None

    def _has_medical_context(self, message_lower):
        """Check if message has medical context despite inappropriate language"""
        medical_keywords = [
            'pain', 'hurt', 'sick', 'disease', 'symptom', 'medicine', 'doctor',
            'hospital', 'treatment', 'medication', 'health', 'diagnosis'
        ]
        return any(keyword in message_lower for keyword in medical_keywords)

    def _is_completely_non_medical(self, message_lower):
        """Check if message is completely unrelated to medical topics"""
        # Random/nonsense text detection
        words = message_lower.split()
        if len(words) == 1 and len(words[0]) > 8:
            # Single very long word that might be random characters
            consonant_count = sum(1 for char in words[0] if char in 'bcdfghjklmnpqrstvwxyz')
            if consonant_count / len(words[0]) > 0.7:  # Too many consonants
                return True
        
        # Common non-medical topics
        non_medical_topics = [
            'car', 'vehicle', 'driving', 'computer', 'software', 'weather',
            'sports', 'music', 'movie', 'food', 'cooking', 'travel',
            'work', 'job', 'school', 'homework', 'money', 'shopping'
        ]
        
        # If message contains non-medical topics and no medical context
        has_non_medical = any(topic in message_lower for topic in non_medical_topics)
        has_medical = self._has_medical_context(message_lower)
        
        return has_non_medical and not has_medical
    
    def find_best_response_pytorch(self, question, confidence_threshold=0.5):
        """Use PyTorch BERT model to classify the question, then find appropriate responses from dataset"""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # STEP 1: Use model to classify the question (not combined text)
            encoding = self.pytorch_tokenizer(
                question,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get model prediction for the question
            with torch.no_grad():
                outputs = self.pytorch_model(input_ids, attention_mask)
                predictions = F.softmax(outputs, dim=1)
                confidence = float(torch.max(predictions))
                predicted_class_idx = torch.argmax(predictions, dim=1).item()
            
            # Always return confidence, even if low
            if confidence < confidence_threshold:
                return None, confidence, "low_confidence"
            
            # Validate prediction
            if predicted_class_idx >= len(self.pytorch_label_encoder.classes_):
                return None, confidence, "unknown_label"
            
            # Convert to actual condition name
            try:
                predicted_condition = self.pytorch_label_encoder.inverse_transform([predicted_class_idx])[0]
            except Exception as e:
                return None, confidence, f"label_error_{predicted_class_idx}"
            
            # STEP 2: Get responses from dataset for this predicted condition
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train_data_chatbot.csv')
            df = pd.read_csv(data_path)
            df_clean = df.dropna(subset=['short_question', 'label', 'short_answer'])
            
            # Find responses for the predicted condition
            condition_responses = df_clean[df_clean['label'] == predicted_condition]
            
            if len(condition_responses) == 0:
                # Try alternative matching if exact label fails
                condition_responses = self.find_alternative_responses_pytorch(predicted_condition, df_clean, predicted_class_idx)
            
            # STEP 3: Select best response from matching condition data
            if not condition_responses.empty:
                response = self.select_diverse_response_pytorch(question, condition_responses, confidence, confidence_threshold)
                return response, confidence, predicted_condition
            else:
                # No responses found - return None but keep confidence info
                return None, confidence, predicted_condition
            
        except Exception as e:
            logger.error(f"Error in PyTorch prediction: {str(e)}")
            return None, 0.0, "prediction_error"
    
    def find_alternative_responses_pytorch(self, predicted_condition, df, label_id):
        """Find responses when exact label match fails for PyTorch"""
        try:
            # Method 1: Try by encoded label if available
            if 'label_encoded' in df.columns:
                alt_responses = df[df['label_encoded'] == label_id]
                if not alt_responses.empty:
                    return alt_responses
        except:
            pass
        
        # Method 2: Partial string matching
        alt_responses = df[df['label'].str.contains(str(predicted_condition), case=False, na=False)]
        if not alt_responses.empty:
            return alt_responses
        
        # Method 3: Return first few rows as fallback
        return df.head(5)
    
    def select_diverse_response_pytorch(self, question, condition_responses, confidence, min_conf=0.5):
        """Select appropriate response from dataset for the predicted condition (PyTorch)"""
        import random
        
        if confidence < min_conf:
            available_responses = condition_responses['short_answer'].tolist()
            if available_responses:
                base_response = random.choice(available_responses)
                return f"Low confidence prediction. Regarding your question: {base_response} However, please consult a healthcare professional for proper diagnosis."
            else:
                return "Low confidence prediction. Please consult a healthcare professional."
        
        # High confidence - select best matching response from dataset
        question_words = set(question.lower().split())
        scored_responses = []
        
        for idx, row in condition_responses.iterrows():
            response_text = str(row['short_answer'])
            dataset_question = str(row['short_question']).lower()
            dataset_words = set(dataset_question.split())
            
            # Calculate similarity score
            overlap = len(question_words.intersection(dataset_words))
            similarity_score = overlap / len(question_words) if question_words else 0
            
            scored_responses.append({
                'response': response_text,
                'score': similarity_score,
                'original_question': row['short_question']
            })
        
        # Sort by similarity and select best match
        scored_responses.sort(key=lambda x: x['score'], reverse=True)
        
        if scored_responses:
            best_response = scored_responses[0]['response']
            return best_response
        else:
            available_responses = condition_responses['short_answer'].tolist()
            if available_responses:
                return random.choice(available_responses)
            else:
                return "Model classified the condition but no specific guidance available."
    
    def find_best_response_tensorflow(self, question, confidence_threshold=0.5):
        """Use TensorFlow model to classify the question, then find appropriate responses from dataset"""
        try:
            # STEP 1: Use model to classify the question (not combined text)
            MAX_LENGTH = 128
            question_seq = self.tf_tokenizer.texts_to_sequences([str(question)])
            question_pad = pad_sequences(question_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
            
            # Get model prediction for the question
            predictions = self.tf_model.predict(question_pad, verbose=0)
            confidence = float(np.max(predictions[0]))
            label_id = np.argmax(predictions[0])
            
            # Always return confidence, even if low
            if confidence < confidence_threshold:
                return None, confidence, "low_confidence"
            
            # Validate prediction
            if label_id >= len(self.tf_label_encoder.classes_):
                return None, confidence, "unknown_label"
            
            # Convert to actual condition name
            try:
                predicted_condition = self.tf_label_encoder.inverse_transform([label_id])[0]
            except Exception as e:
                return None, confidence, f"label_error_{label_id}"
            
            # STEP 2: Get responses from dataset for this predicted condition
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train_data_chatbot.csv')
            df = pd.read_csv(data_path)
            df_clean = df.dropna(subset=['short_question', 'label', 'short_answer'])
            
            # Find responses for the predicted condition
            condition_responses = df_clean[df_clean['label'] == predicted_condition]
            
            if len(condition_responses) == 0:
                # Try alternative matching if exact label fails
                condition_responses = self.find_alternative_responses_tf(predicted_condition, df_clean, label_id)
            
            # STEP 3: Select best response from matching condition data
            if not condition_responses.empty:
                response = self.select_diverse_response_tf(question, condition_responses, confidence, confidence_threshold)
                return response, confidence, predicted_condition
            else:
                # No responses found - return None but keep confidence info
                return None, confidence, predicted_condition
            
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            return None, 0.0, "prediction_error"
    
    def find_alternative_responses_tf(self, predicted_condition, df, label_id):
        """Find responses when exact label match fails"""
        try:
            # Method 1: Try by encoded label if available
            if 'label_encoded' in df.columns:
                alt_responses = df[df['label_encoded'] == label_id]
                if not alt_responses.empty:
                    return alt_responses
        except:
            pass
        
        # Method 2: Partial string matching
        alt_responses = df[df['label'].str.contains(str(predicted_condition), case=False, na=False)]
        if not alt_responses.empty:
            return alt_responses
        
        # Method 3: Return first few rows as fallback
        return df.head(5)
    
    def select_diverse_response_tf(self, question, condition_responses, confidence, min_conf=0.5):
        """Select appropriate response from dataset for the predicted condition"""
        import random
        
        if confidence < min_conf:
            available_responses = condition_responses['short_answer'].tolist()
            if available_responses:
                base_response = random.choice(available_responses)
                return f"Low confidence prediction. Regarding your question: {base_response} However, please consult a healthcare professional for proper diagnosis."
            else:
                return "Low confidence prediction. Please consult a healthcare professional."
        
        # High confidence - select best matching response from dataset
        question_words = set(question.lower().split())
        scored_responses = []
        
        for idx, row in condition_responses.iterrows():
            response_text = str(row['short_answer'])
            dataset_question = str(row['short_question']).lower()
            dataset_words = set(dataset_question.split())
            
            # Calculate similarity score
            overlap = len(question_words.intersection(dataset_words))
            similarity_score = overlap / len(question_words) if question_words else 0
            
            scored_responses.append({
                'response': response_text,
                'score': similarity_score,
                'original_question': row['short_question']
            })
        
        # Sort by similarity and select best match
        scored_responses.sort(key=lambda x: x['score'], reverse=True)
        
        if scored_responses:
            best_response = scored_responses[0]['response']
            return best_response
        else:
            available_responses = condition_responses['short_answer'].tolist()
            if available_responses:
                return random.choice(available_responses)
            else:
                return "Model classified the condition but no specific guidance available."
    
    def predict_tensorflow(self, question):
        """Make prediction using TensorFlow model with comprehensive input handling"""
        # Check for emergency first (highest priority)
        if self.detect_emergency(question):
            return "🚨 MEDICAL EMERGENCY DETECTED! 🚨\n\nCall 999 or your local emergency number IMMEDIATELY!\n\nFor chest pain, breathing difficulties, stroke symptoms, severe bleeding, or any life-threatening condition, emergency services should be contacted right away.\n\nThis is not a substitute for emergency medical care."
        
        # Check for inappropriate language or complaints
        inappropriate_type = self.detect_inappropriate_language(question)
        if inappropriate_type == 'complaint':
            return "I understand you may be frustrated, and I apologize if my responses haven't met your expectations. I'm here to provide helpful medical information to the best of my ability. Please let me know how I can better assist you with your health-related questions. Your feedback is valuable for improving my service."
        elif inappropriate_type == 'inappropriate':
            return "I'm here to provide professional medical assistance. Please feel free to ask any health-related questions you may have, and I'll do my best to help you with accurate and helpful information."
        elif inappropriate_type == 'non_medical':
            return "I'm a medical chatbot designed to help with health-related questions. I may not be able to assist with topics outside of medicine and healthcare. Please feel free to ask me about any medical concerns, symptoms, treatments, or health conditions you'd like to know about."
        
        # Check for greeting
        greeting_response = self.detect_greeting(question)
        if greeting_response:
            return greeting_response
            
        if not self.is_tf_loaded:
            return "Sorry, the TensorFlow model is not available at the moment. Please try again later or contact support if the issue persists."
        
        try:
            # Find the best response using proper classification approach
            best_response, best_score, best_label = self.find_best_response_tensorflow(question)
            
            # Check if we got a valid response with sufficient confidence
            if best_response and best_score and best_score > 0.5:
                return f"{best_response}\n\n(Confidence: {best_score:.2%} | Category: {best_label})"
            else:
                # Low confidence explanation for ALL unsupported intents
                confidence_info = f" (Confidence: {best_score:.2%})" if best_score else ""
                return f"I'm not confident enough to provide a specific answer for your question{confidence_info}. This could be because:\n• Your question might be outside my medical knowledge base\n• The symptoms described are unclear or too general\n• Your question might need personalized medical assessment\n• The input may not be a recognizable medical query\n\nFor the most accurate diagnosis and treatment advice, please consult with a qualified healthcare professional."
        
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            return "Sorry, I encountered a technical error while processing your question. Please try rephrasing your question or try again later."
    
    def predict_pytorch(self, question):
        """Make prediction using PyTorch BERT model with comprehensive input handling"""
        # Check for emergency first (highest priority)
        if self.detect_emergency(question):
            return "🚨 MEDICAL EMERGENCY DETECTED! 🚨\n\nCall 999 or your local emergency number IMMEDIATELY!\n\nFor chest pain, breathing difficulties, stroke symptoms, severe bleeding, or any life-threatening condition, emergency services should be contacted right away.\n\nThis is not a substitute for emergency medical care."
        
        # Check for inappropriate language or complaints
        inappropriate_type = self.detect_inappropriate_language(question)
        if inappropriate_type == 'complaint':
            return "I understand you may be frustrated, and I apologize if my responses haven't met your expectations. I'm here to provide helpful medical information to the best of my ability. Please let me know how I can better assist you with your health-related questions. Your feedback is valuable for improving my service."
        elif inappropriate_type == 'inappropriate':
            return "I'm here to provide professional medical assistance. Please feel free to ask any health-related questions you may have, and I'll do my best to help you with accurate and helpful information."
        elif inappropriate_type == 'non_medical':
            return "I'm a medical chatbot designed to help with health-related questions. I may not be able to assist with topics outside of medicine and healthcare. Please feel free to ask me about any medical concerns, symptoms, treatments, or health conditions you'd like to know about."
        
        # Check for greeting
        greeting_response = self.detect_greeting(question)
        if greeting_response:
            return greeting_response
            
        if not self.is_pytorch_loaded:
            return "Sorry, the PyTorch BERT model is not available at the moment. Please try again later or contact support if the issue persists."
        
        try:
            # Find the best response using proper classification approach
            best_response, best_score, best_label = self.find_best_response_pytorch(question)
            
            # Check if we got a valid response with sufficient confidence
            if best_response and best_score and best_score > 0.5:
                return f"{best_response}\n\n(BERT Confidence: {best_score:.2%} | Category: {best_label})"
            else:
                # Low confidence explanation for ALL unsupported intents
                confidence_info = f" (BERT Confidence: {best_score:.2%})" if best_score else ""
                return f"I'm not confident enough to provide a specific answer for your question{confidence_info}. This could be because:\n• Your question might be outside my medical knowledge base\n• The symptoms described are unclear or too general\n• Your question might need personalized medical assessment\n• The input may not be a recognizable medical query\n\nFor the most accurate diagnosis and treatment advice, please consult with a qualified healthcare professional."
        
        except Exception as e:
            logger.error(f"Error in PyTorch prediction: {str(e)}")
            return "Sorry, I encountered a technical error while processing your question. Please try rephrasing your question or try again later."

# Initialize the chatbot
chatbot = MedicalChatbot()

@app.route('/')
def index():
    """Serve the main model selection interface"""
    return render_template('index.html')

@app.route('/tensorflow')
def tensorflow_chat():
    """Serve the TensorFlow chat interface"""
    return render_template('tensorflow.html')

@app.route('/pytorch')
def pytorch_chat():
    """Serve the PyTorch BERT chat interface"""
    return render_template('pytorch.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        model_type = data.get('model', 'tensorflow').lower()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty'
            })
        
        # Generate response based on selected model
        if model_type == 'tensorflow':
            response = chatbot.predict_tensorflow(message)
        elif model_type == 'pytorch':
            response = chatbot.predict_pytorch(message)
        else:
            response = "Invalid model type selected."
        
        return jsonify({
            'success': True,
            'response': response,
            'model_used': model_type,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your request.'
        })

@app.route('/api/models/status', methods=['GET'])
def model_status():
    """Get the status of available models"""
    return jsonify({
        'tensorflow': {
            'loaded': chatbot.is_tf_loaded,
            'status': 'Ready' if chatbot.is_tf_loaded else 'Error loading model'
        },
        'pytorch': {
            'loaded': chatbot.is_pytorch_loaded,
            'status': 'Training in progress' if not chatbot.is_pytorch_loaded else 'Ready'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_available': {
            'tensorflow': chatbot.is_tf_loaded,
            'pytorch': chatbot.is_pytorch_loaded
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
