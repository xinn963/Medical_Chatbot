========================================================================
                          MEDICAL CHATBOT PROJECT
                            USER GUIDE & INSTRUCTIONS
========================================================================

PROJECT OVERVIEW
================================================================================

This is an AI-powered medical chatbot system that can answer health-related 
questions using two different machine learning approaches:

- TensorFlow Model: Traditional BiLSTM neural network approach
- PyTorch Model: Advanced BERT transformer approach

IMPORTANT DISCLAIMER: This system is for educational purposes only. 
Always consult real healthcare professionals for actual medical advice.

PROJECT STRUCTURE
================================================================================

Medical_Chatbot1/
├── data/
│   └── train_data_chatbot.csv          (Training dataset - 48,245 Q&A pairs)
├── tensorflow_version/
│   ├── TensorFlow.ipynb                (Training notebook for TensorFlow model)
│   ├── app.py                          (Flask app for TensorFlow only)
│   ├── medical_tokenizer.pkl           (Text processing component)
│   ├── medical_label_encoder.pkl       (Category mapping component)
│   └── medical_models/
│       └── best_medical_model.weights.h5  (Trained TensorFlow model)
├── pytorch_version/
│   ├── pytorch.ipynb                   (Training notebook for PyTorch model)
│   ├── medical_label_encoder.pkl       (Category mapping component)
│   └── medical_models/
│       └── bert_best_model.pt          (Trained PyTorch BERT model)
├── web_ui/
│   ├── app.py                          (Main Flask application - DUAL MODEL)
│   ├── static/                         (CSS and JavaScript files)
│   └── templates/
│       └── index.html                  (Web interface HTML)
└── README.txt                          (This file)

SYSTEM REQUIREMENTS
================================================================================

- Python 3.7 or higher
- Jupyter Notebook
- Required Python packages:
  * TensorFlow 2.x
  * PyTorch
  * Transformers (Hugging Face)
  * Flask
  * scikit-learn
  * pandas
  * numpy
  * pickle

HOW TO START THE SYSTEM
================================================================================

OPTION A: QUICK START (Using Pre-trained Models)
------------------------------------------------
If model files already exist, follow these steps:

1. Open Command Prompt or Terminal
2. Navigate to the web_ui folder:
   cd "C:\Users\..\web_ui" (the path in your computer)
3. Run the Flask application:
   python app.py
4. Wait for loading messages:
   "✅ TensorFlow model loaded successfully"
   "✅ PyTorch BERT model loaded successfully"
5. Open web browser and go to: http://localhost:5000
6. Choose either TensorFlow or PyTorch model
7. Start asking medical questions

OPTION B: FULL TRAINING PROCESS (Training from Scratch)
-------------------------------------------------------
If you need to train the models first:

STEP 1: Train TensorFlow Model
1. Open Jupyter Notebook: jupyter notebook
2. Navigate to tensorflow_version/TensorFlow.ipynb
3. Run all cells from top to bottom (1-2 hours)
4. Wait for "✅ Medical model training complete!" message
5. Verify file created: medical_models/best_medical_model.weights.h5

STEP 2: Train PyTorch Model
1. Navigate to pytorch_version/pytorch.ipynb
2. Run all cells from top to bottom (2-3 hours)
3. Wait for "✅ PyTorch BERT model loaded successfully" message
4. Verify file created: medical_models/bert_best_model.pt

STEP 3: Verify Required Files Exist
- tensorflow_version/medical_tokenizer.pkl
- tensorflow_version/medical_label_encoder.pkl
- tensorflow_version/medical_models/best_medical_model.weights.h5
- pytorch_version/medical_label_encoder.pkl
- pytorch_version/medical_models/bert_best_model.pt

STEP 4: Start Web Application
1. Open Command Prompt or Terminal
2. Navigate to web_ui folder:
   cd "C:\Users\..\web_ui" (the path in your computer)
3. Run: python app.py
4. Open browser: http://localhost:5000

HOW TO USE THE SYSTEM
================================================================================

ACCESSING THE WEB INTERFACE
1. Open web browser
2. Go to: http://localhost:5000 or http://127.0.0.1:5000
3. You'll see the main page with two model options

CHOOSING AI MODEL
- Click "TensorFlow Chat" for BiLSTM neural network approach
- Click "PyTorch Chat" for BERT transformer approach
- Both models use the same medical dataset but different AI architectures

ASKING QUESTIONS
- Type your medical question in the chat box
- Press Enter or click Send button
- AI will analyze and provide medical information
- Response includes confidence level and medical category

EXAMPLE QUESTIONS
- "Can antibiotics cause a rash?"
- "What are the symptoms of diabetes?"
- "I have a headache and feel tired"
- "Can I take ibuprofen with high blood pressure?"
- "What causes chest pain?"

SAFETY FEATURES
================================================================================

EMERGENCY DETECTION
The system detects emergency keywords such as:
- "severe chest pain"
- "difficulty breathing"
- "stroke symptoms"
- "heart attack"
- "unconscious"

Response: "🚨 MEDICAL EMERGENCY! Call 999 immediately!"

SMART RESPONSE HANDLING
- Medical questions: Provides relevant medical information
- Greetings: Responds as friendly medical assistant
- Inappropriate content: Professional redirection to medical topics
- Unknown topics: Explains low confidence and advises consulting professionals

SYSTEM INTERFACE
================================================================================

MAIN PAGE FEATURES
- Clean, professional medical-themed design
- Model selection screen (TensorFlow vs PyTorch)
- Welcome message and medical disclaimers
- Easy navigation between different AI models

CHAT INTERFACE FEATURES
- Real-time messaging system
- Confidence scores displayed for each response
- Medical category classification shown
- Professional styling with medical color scheme
- Clear question/answer format

TROUBLESHOOTING
================================================================================

IF SYSTEM WON'T START:
1. Check you're in correct directory (web_ui folder)
2. Verify Python is properly installed
3. Install missing packages: pip install tensorflow torch transformers flask
4. Look for error messages in terminal/command prompt

IF MODELS DON'T LOAD:
1. Verify all required files exist (see file list above)
2. Check file paths are correct
3. Ensure model training completed successfully
4. Re-run training notebooks if files are missing

IF RESPONSES SEEM INCORRECT:
1. Try rephrasing your question more specifically
2. Use proper medical terminology when possible
3. Ask one question at a time
4. Check confidence score - low confidence means uncertain prediction

TECHNICAL SPECIFICATIONS
================================================================================

TENSORFLOW MODEL ARCHITECTURE
- Embedding layer (10,000 vocabulary, 100 dimensions)
- Bidirectional LSTM (128 units)
- Dense layers with dropout for classification
- Softmax output for 48 medical categories

PYTORCH MODEL ARCHITECTURE  
- BERT base model (bert-base-uncased)
- Additional classification head
- Fine-tuned on medical dataset
- Transformer-based attention mechanism

DATASET INFORMATION
- 48,245 medical question-answer pairs
- Covers various medical topics and conditions
- Preprocessed and labeled for supervised learning
- Split into training and validation sets

WEB APPLICATION FRAMEWORK
- Flask backend server
- HTML/CSS/JavaScript frontend
- REST API for model predictions
- Real-time chat interface

IMPORTANT REMINDERS
================================================================================

MEDICAL DISCLAIMER
- This AI provides information, NOT medical diagnosis
- Emergency situations: Call 999 or local emergency number
- Serious symptoms: Always consult healthcare professionals
- Educational use: System designed for learning about AI in healthcare

USAGE TIPS FOR BETTER RESULTS
- Be specific: "headache after exercise" vs "I feel bad"
- Use medical terms: "nausea" vs "feeling sick"  
- Ask one question at a time
- Check confidence scores: Higher confidence = more reliable response

PROJECT SUBMISSION NOTES
================================================================================

FOR INSTRUCTORS/TUTORS:
This project demonstrates:
- Implementation of two different AI architectures (BiLSTM vs BERT)
- Medical dataset processing and training
- Web application development with Flask
- Safety features for medical AI systems
- Complete end-to-end ML pipeline from data to deployment

DEMONSTRATION WORKFLOW:
1. Show model training in Jupyter notebooks (optional if pre-trained)
2. Start web application from command line
3. Demonstrate both TensorFlow and PyTorch models
4. Test various question types and safety features
5. Explain confidence scoring and response categorization

========================================================================
END OF DOCUMENTATION
========================================================================