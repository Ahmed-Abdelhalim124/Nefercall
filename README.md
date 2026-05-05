#  Nefercall 

A bilingual (English/Arabic) voice-enabled banking customer service agent powered by Retrieval-Augmented Generation (RAG), featuring real-time speech recognition, natural language understanding, and text-to-speech capabilities.

##  Key Features

###  **Advanced Voice Processing**
- **Dual-Language Support**: English and Arabic voice recognition
- **Smart VAD (Voice Activity Detection)**: Automatic speech detection with silence monitoring
- **Real-time Transcription**: Powered by Whisper Large V3 for English and Whisper Turbo for Arabic
- **Natural TTS**: High-quality text-to-speech in both languages

###  **Intelligent RAG Pipeline**
- **Query Expansion**: Enhanced retrieval with automatic query variations
- **Semantic Search**: FAISS-powered vector search with BGE embeddings
- **Cross-Encoder Reranking**: Precision ranking using MS-MARCO MiniLM
- **Multi-Intent Recognition**: Handles complex queries with multiple requests
- **Zero-Hallucination Architecture**: Strict response validation and confidence scoring

###  **Conversational Intelligence**
- **Soft Query Handling**: Natural responses to greetings, thanks, and farewells
- **Context-Aware Responses**: Maintains conversation history across turns
- **Dynamic Greeting Generation**: Personalized welcome messages
- **Session Management**: Redis-backed persistent sessions with automatic expiry

###  **Enterprise-Ready**
- **Confidence Thresholds**: Configurable validation for single and multi-intent queries
- **Fallback Mechanisms**: Graceful handling of low-confidence scenarios
- **Memory Management**: Efficient conversation history with configurable limits
- **Error Resilience**: Comprehensive error handling and logging


##  Dataset

The system uses the **Bitext Retail Banking LLM Chatbot Training Dataset** from Hugging Face:

- **Source**: `bitext/Bitext-retail-banking-llm-chatbot-training-dataset`
- **Size**: 27,000+ banking customer service examples
- **Coverage**: Multiple intents and categories including:
  - Account management
  - Card services
  - Payments and transfers
  - Customer support
  - Security and fraud

##  Usage

### Starting a Call

1. **Select Language**: Choose English 🇬🇧 or Arabic 🇸🇦
2. **Click "Start Call"**: Grants microphone access
3. **Listen to Greeting**: Agent welcomes you
4. **Speak Your Query**: Ask banking questions naturally
5. **Get Response**: Receive accurate answers

### Example Queries

**English:**
- "What's my account balance?"
- "How do I transfer money to another person?"
- "I lost my credit card, what should I do?"

**Arabic:**
- "ما هو رصيد حسابي؟"
- "كيف أحول المال إلى شخص آخر؟"
- "فقدت بطاقتي الائتمانية، ماذا أفعل؟"



##  Hints


**"Redis not available, using in-memory"**
- Redis is optional; the system will use in-memory storage
- For production, will install Redis for persistence

**"Audio playback error"**
- Check browser audio permissions
- Ensure audio codec support (MP3/WAV)

**"WebSocket disconnected"**
- Check network connectivity
- Verify firewall settings for WebSocket connections


**Low transcription accuracy**
- Speak clearly and at moderate pace
- Reduce background noise
- Check microphone quality


## Prices

**"Average Customer Service Call Cost (5 turns)"**
| API                                    |           Tokens Used | Cost per Call |
| -------------------------------------- | --------------------: | ------------: |
| **Groq (Llama 3.3 70B)**               |            8,500 text |  **~$0.0059** |
| **gemini-2.5-flash (STT + TTS)**       | 13,100 (text + audio) |  **~$0.0210** |
