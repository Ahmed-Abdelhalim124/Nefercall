#!/usr/bin/env python3

from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from typing import List, Dict, Optional, Tuple
from groq import Groq
import warnings
import json
import uuid
from datetime import datetime
import io
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import redis

warnings.filterwarnings('ignore')

GROQ_API_KEY = "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"
HOST = "localhost"
PORT = 8000
REDIS_HOST = "localhost"
REDIS_PORT = 6379

class BilingualVoiceBankingRAG:
    def __init__(self, groq_api_key: str):
        print("="*80)
        print("🌍 BILINGUAL VOICE BANKING RAG - FULL PIPELINE (Enhanced)")
        print("="*80)
        
        self._init_redis()
        self._init_groq(groq_api_key)
        self._load_dataset()
        self._prepare_knowledge_base()
        self._load_models()
        self._create_embeddings()
        self._build_faiss_index()
        self._set_config()
        self._init_memory_sessions()
        
        print("\n✅ System Ready!")
        print("="*80 + "\n")
    
    def _init_redis(self):
        print("\n[Redis] Connecting to Redis...")
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0,
                decode_responses=True, socket_connect_timeout=5
            )
            self.redis_client.ping()
            print(f"✓ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
            self.use_redis = True
        except Exception as e:
            print(f"⚠️ Redis not available, using in-memory: {e}")
            self.use_redis = False
    
    def _init_groq(self, groq_api_key: str):
        print("\n[1/6] Connecting to Groq API...")
        self.groq_client = Groq(api_key=groq_api_key)
        print("✓ Connected")
    
    def _load_dataset(self):
        print("\n[2/6] Loading Banking Dataset...")
        self.dataset = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset")["train"]
        print(f"✓ Loaded {len(self.dataset)} examples")
    
    def _prepare_knowledge_base(self):
        print("\n[3/6] Preparing knowledge base...")
        self.knowledge_docs = []
        self.query_texts = []
        self.metadata = []
        self.intent_index = {}
        self.category_index = {}
        
        for idx, row in enumerate(self.dataset):
            doc = f"Intent: {row['intent']}\nCategory: {row['category']}\nCustomer Query: {row['instruction']}\nAgent Response: {row['response']}"
            self.knowledge_docs.append(doc)
            self.query_texts.append(row['instruction'])
            
            self.metadata.append({
                'idx': idx,
                'intent': row['intent'],
                'category': row['category'],
                'instruction': row['instruction'],
                'response': row['response']
            })
            
            if row['intent'] not in self.intent_index:
                self.intent_index[row['intent']] = []
            self.intent_index[row['intent']].append(idx)
            
            if row['category'] not in self.category_index:
                self.category_index[row['category']] = []
            self.category_index[row['category']].append(idx)
        
        print(f"✓ Created {len(self.knowledge_docs)} documents")
        print(f"✓ Intent types: {len(self.intent_index)}")
        print(f"✓ Categories: {len(self.category_index)}")
    
    def _load_models(self):
        print("\n[4/6] Loading models...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✓ Models loaded")
    
    def _create_embeddings(self):
        print("\n[5/6] Creating embeddings...")
        self.query_embeddings = self.embedding_model.encode(
            self.query_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        print(f"✓ Embeddings shape: {self.query_embeddings.shape}")
    
    def _build_faiss_index(self):
        print("\n[6/6] Building FAISS index...")
        self.dimension = self.query_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(self.query_embeddings)
        self.index.add(self.query_embeddings)
        print(f"✓ Index built: {self.index.ntotal} vectors")
    
    def _set_config(self):
        self.CONFIDENCE_THRESHOLDS = {
            'single_intent': {
                'high_rerank': 2.0,
                'medium_rerank': 0.5,
                'low_rerank': -1.0,
                'min_semantic': 0.20
            },
            'multi_intent': {
                'high_rerank': 1.5,
                'medium_rerank': 0.0,
                'low_rerank': -2.0,
                'min_semantic': 0.15
            }
        }
        self.MAX_HISTORY_LENGTH = 10
        self.SESSION_EXPIRY = 3600  
    
    def _init_memory_sessions(self):
        self.memory_sessions = {}
    
    
    def create_session(self, initial_language: str = 'en') -> str:
        session_id = str(uuid.uuid4())
        session_data = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'conversation_history': [],
            'language': initial_language
        }
        if self.use_redis:
            try:
                self.redis_client.setex(f"session:{session_id}", self.SESSION_EXPIRY, json.dumps(session_data))
            except:
                self.memory_sessions[session_id] = session_data
        else:
            self.memory_sessions[session_id] = session_data
        print(f"✅ Session created: {session_id[:8]}... (Language: {initial_language})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        if self.use_redis:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return json.loads(data)
            except:
                pass
        return self.memory_sessions.get(session_id)
    
    def update_session(self, session_id: str, session_data: Dict):
        if self.use_redis:
            try:
                self.redis_client.setex(f"session:{session_id}", self.SESSION_EXPIRY, json.dumps(session_data))
                return
            except:
                pass
        self.memory_sessions[session_id] = session_data
    
    def add_to_history(self, session_id: str, turn_data: Dict):
        session = self.get_session(session_id)
        if session:
            session['conversation_history'].append({
                'timestamp': datetime.now().isoformat(),
                'customer_query': turn_data['query'],
                'agent_response': turn_data['response'],
                'language': turn_data.get('language', 'en'),
                'intents': turn_data.get('intents', [])
            })
            if len(session['conversation_history']) > self.MAX_HISTORY_LENGTH:
                session['conversation_history'] = session['conversation_history'][-self.MAX_HISTORY_LENGTH:]
            self.update_session(session_id, session)
    
    def end_session(self, session_id: str) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        summary = {
            'session_id': session_id,
            'duration': (datetime.now() - datetime.fromisoformat(session['created_at'])).total_seconds(),
            'total_turns': len(session['conversation_history']),
            'language': session['language']
        }
        
        if self.use_redis:
            try:
                self.redis_client.delete(f"session:{session_id}")
            except:
                pass
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
        
        print(f"🔚 Session ended: {session_id[:8]}... ({summary['total_turns']} turns)")
        return summary
    
    
    def _detect_language(self, text: str) -> str:
        try:
            prompt = f"""Detect the language of this text. Respond with ONLY 'ENGLISH' or 'ARABIC'.
Text: "{text}"
Language:"""
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            return 'ar' if 'ARABIC' in result else 'en'
        except:
            return 'en'
    
    def _translate_to_english(self, text: str) -> str:
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": f"Translate this Arabic text to English, output only the translation:\n{text}"}],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return text
    
    def _translate_response_to_arabic(self, english_response: str) -> str:
        try:
            prompt = f"""Translate this banking response to Arabic MSA. Output ONLY the translation, no preamble:

{english_response}"""
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except:
            return english_response
    
    def _generate_dynamic_greeting(self, language: str) -> str:
        try:
            if language == 'ar':
                prompt = "أنشئ تحية بنكية دافئة ومهنية بالعربية الفصحى (جملة واحدة فقط)"
            else:
                prompt = "Generate a warm, professional banking greeting (1 sentence only)"
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.8,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except:
            return "مرحباً بك! كيف يمكنني مساعدتك اليوم؟" if language == 'ar' else "Hello! How can I help you today?"
    
    
    def transcribe_audio(self, audio_bytes: bytes, language: str) -> str:
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            with open(tmp_path, 'rb') as f:
                model = "whisper-large-v3" if language == 'ar' else "whisper-large-v3-turbo"
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(tmp_path, f.read()),
                    model=model,
                    language="ar" if language == 'ar' else "en",
                    response_format="text",
                    temperature=0.0
                )
            
            os.unlink(tmp_path)
            
            text = transcription.strip() if isinstance(transcription, str) else ""
            
            noise_phrases = ["thank you for watching", "subscribe", ".", "...", "you", "thank you"]
            if text.lower() in noise_phrases or len(text) < 3:
                return ""
            
            print(f"🎤 Transcribed ({language}): {text}")
            return text
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    def text_to_speech(self, text: str, language: str = 'en') -> bytes:
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='ar' if language == 'ar' else 'en', slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return b""
    
    
    def _expand_query_enhanced(self, query: str) -> List[str]:
        try:
            prompt = f"""Generate 3 alternative phrasings:
Original: {query}
Provide ONLY 3 alternatives, one per line:"""
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=150
            )
            variations = [v.strip() for v in response.choices[0].message.content.strip().split('\n') if v.strip()]
            cleaned = []
            for v in variations[:3]:
                if v and len(v) > 5:
                    if v[0].isdigit() and v[1] in '.-)':
                        v = v[2:].strip()
                    cleaned.append(v)
            return [query] + cleaned
        except:
            return [query]
    
    def retrieve_semantic_enhanced(self, query: str, k: int = 20) -> List[Dict]:
        queries = self._expand_query_enhanced(query)
        all_results = []
        for q in queries:
            q_embedding = self.embedding_model.encode([q], convert_to_numpy=True)
            faiss.normalize_L2(q_embedding)
            scores, indices = self.index.search(q_embedding, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.1:
                    all_results.append({
                        'idx': int(idx),
                        'score': float(score),
                        'query_variant': q
                    })
        seen = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x['score'], reverse=True):
            if r['idx'] not in seen:
                seen.add(r['idx'])
                unique_results.append(r)
        return unique_results[:k]
    
    def rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        pairs = []
        for cand in candidates:
            idx = cand['idx']
            instruction = self.metadata[idx]['instruction']
            pairs.append([query, instruction])
        rerank_scores = self.reranker.predict(pairs)
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = float(rerank_scores[i])
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    def retrieve_for_intent(self, intent_query: str, top_k: int = 10) -> List[Dict]:
        """Enhanced retrieval with query expansion"""
        semantic_results = self.retrieve_semantic_enhanced(intent_query, k=20)
        candidates = [{'idx': r['idx'], 'semantic_score': r['score']} for r in semantic_results]
        reranked = self.rerank_results(intent_query, candidates[:15])
        
        results = []
        for r in reranked[:top_k]:
            idx = r['idx']
            results.append({
                'document': self.knowledge_docs[idx],
                'metadata': self.metadata[idx],
                'semantic_score': r.get('semantic_score', 0.0),
                'rerank_score': r.get('rerank_score', 0.0),
                'final_score': r.get('rerank_score', 0.0)
            })
        return results
    
    def validate_retrieval(self, retrieved_docs: List[Dict], intent_type: str = 'single_intent') -> Dict:
        if not retrieved_docs:
            return {
                'can_answer': False,
                'confidence': 'none',
                'reason': 'No documents retrieved'
            }
        
        best_doc = retrieved_docs[0]
        rerank_score = best_doc['final_score']
        semantic_score = best_doc['semantic_score']
        
        thresholds = self.CONFIDENCE_THRESHOLDS[intent_type]
        
        if rerank_score >= thresholds['high_rerank'] and semantic_score >= thresholds['min_semantic']:
            confidence = 'high'
            can_answer = True
        elif rerank_score >= thresholds['medium_rerank'] and semantic_score >= thresholds['min_semantic']:
            confidence = 'medium'
            can_answer = True
        elif rerank_score >= thresholds['low_rerank'] and semantic_score >= thresholds['min_semantic']:
            confidence = 'low'
            can_answer = True
        else:
            confidence = 'very_low'
            can_answer = False
        
        return {
            'can_answer': can_answer,
            'confidence': confidence,
            'best_rerank_score': rerank_score,
            'best_semantic_score': semantic_score,
            'reason': f"Rerank: {rerank_score:.2f}, Semantic: {semantic_score:.2f}"
        }
    
    
    def _classify_query_type(self, query: str) -> str:
        """Classify if query is conversational/soft or banking-specific"""
        try:
            prompt = f"""Classify this customer message into ONE category:

GREETING: Hello, hi, good morning, how are you, etc.
THANKS: Thank you, thanks, appreciate it, etc.
FAREWELL: Goodbye, bye, see you, have a nice day, etc.
CLARIFICATION: What?, Can you repeat?, I don't understand, etc.
AFFIRMATION: Yes, okay, sure, I see, got it, etc.
BANKING: Any actual banking question or request

Message: "{query}"
Category:"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=20
            )
            
            category = response.choices[0].message.content.strip().upper()
            
            if any(word in category for word in ['GREETING', 'THANKS', 'FAREWELL', 'CLARIFICATION', 'AFFIRMATION']):
                return category.split(':')[0].strip()
            else:
                return 'BANKING'
                
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return 'BANKING'
    
    def _generate_soft_response(self, query: str, category: str, language: str = 'en') -> str:
        """Generate natural conversational responses with controlled creativity"""
        try:
            if language == 'ar':
                prompt = f"""أنت موظف خدمة عملاء بنكي محترف. الرد على هذه الرسالة بطريقة طبيعية ودافئة.

نوع الرسالة: {category}
رسالة العميل: "{query}"

قواعد:
- رد واحد قصير (1-2 جمل فقط)
- كن ودوداً ومهنياً
- إذا كانت تحية، رحب واسأل كيف يمكنك المساعدة
- إذا كانت شكر، رد بلطف وعرض المزيد من المساعدة
- إذا كانت وداع، ودّع بلطف
- استخدم العربية الفصحى فقط
- لا تضف ملاحظات أو تفسيرات

الرد:"""
            else:
                prompt = f"""You're a professional banking customer service agent. Respond to this message naturally and warmly.

Message Type: {category}
Customer Message: "{query}"

Rules:
- One short response (1-2 sentences only)
- Be friendly and professional
- If greeting, welcome and ask how you can help
- If thanks, respond politely and offer further assistance
- If farewell, say goodbye warmly
- Keep it natural and conversational
- No notes or explanations

Response:"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,  
                max_tokens=100
            )
            
            result = response.choices[0].message.content.strip()
            
            result = result.replace('"', '').strip()
            
            return result
            
        except Exception as e:
            print(f"❌ Soft response generation error: {e}")
            if language == 'ar':
                fallbacks = {
                    'GREETING': 'مرحباً بك! كيف يمكنني مساعدتك اليوم؟',
                    'THANKS': 'على الرحب والسعة! هل تحتاج إلى أي مساعدة أخرى؟',
                    'FAREWELL': 'مع السلامة! نتمنى لك يوماً سعيداً.',
                    'AFFIRMATION': 'ممتاز! كيف يمكنني مساعدتك؟'
                }
            else:
                fallbacks = {
                    'GREETING': 'Hello! How can I assist you today?',
                    'THANKS': 'You\'re welcome! Is there anything else I can help you with?',
                    'FAREWELL': 'Goodbye! Have a great day.',
                    'AFFIRMATION': 'Great! How can I help you?'
                }
            return fallbacks.get(category, fallbacks['GREETING'])
    
    
    def generate_response(self, query: str, intent_results: List[Dict], language: str = 'en') -> str:
        
        answerable_intents = [ir for ir in intent_results if ir['validation']['can_answer']]
        
        if not answerable_intents:
            print(f"❌ Banking query cannot be answered (low confidence)")
            if language == 'ar':
                return "أعتذر، لا أملك معلومات كافية للإجابة على هذا السؤال البنكي. هل يمكنك إعادة صياغته أو طرح سؤال آخر متعلق بالخدمات البنكية المتاحة؟"
            else:
                return "I apologize, but I don't have enough information to answer this banking question. Could you rephrase it or ask about our available banking services?"
        
        context_parts = []
        for intent_result in answerable_intents:
            for doc in intent_result['retrieved_docs'][:2]:
                context_parts.append(
                    f"Q: {doc['metadata']['instruction']}\n"
                    f"A: {doc['metadata']['response']}"
                )
        
        context = "\n\n".join(context_parts)
        
        system_prompt = f"""You are a professional banking customer service agent. CRITICAL RULES:
1. ONLY use information from the reference examples provided below
2. NEVER add information not present in the examples
3. NEVER make up branch locations, addresses, phone numbers, or ANY specific details
4. NEVER use general knowledge about banks or cities
5. If examples say "use our branch locator tool", say exactly that
6. If examples say "visit our website", say exactly that
7. Keep the same level of detail as the examples (generic stays generic)
8. Paraphrase naturally but stay factually identical to examples
9. Be concise, clear, and helpful
10. If multiple requests, address all of them
11. Maintain a professional, friendly tone
12. Respond ONLY in {'Arabic (MSA)' if language == 'ar' else 'English'}
13. NEVER add any notes, explanations, disclaimers, or comments like "Note:", "I've used Modern Standard Arabic", "MSA", "translation", etc.
14. Output MUST be ONLY the final customer-facing response — nothing else."""
        
        user_prompt = f"""Customer Query: "{query}"

Reference Information:
{context}

Generate the response in {'Arabic' if language == 'ar' else 'English'} ONLY.
No notes. No explanations. Just the pure response:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            
            result = result.split("Arabic translation:")[-1].strip()
            result = result.split("Response:")[-1].strip()
            result = result.split("الرد:")[-1].strip()
            
            if language == 'ar':
                result = self._translate_response_to_arabic(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return answerable_intents[0]['retrieved_docs'][0]['metadata']['response'] if answerable_intents else "Technical issue."
    
    
    def _decompose_query(self, query: str) -> Dict:
        try:
            prompt = f"""Analyze this banking customer query and extract all distinct intents/requests.

Query: "{query}"

Output format:
NUMBER_OF_INTENTS: [single number]
INTENT_1: [first specific request]
INTENT_2: [second specific request if exists]
...

Example:
Query: "I want to check my balance and transfer money to my friend"
NUMBER_OF_INTENTS: 2
INTENT_1: Check account balance
INTENT_2: Transfer money to another person"""

            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            lines = content.split('\n')
            
            intents = []
            for line in lines:
                if line.startswith('INTENT_'):
                    intent_text = ':'.join(line.split(':')[1:]).strip()
                    if intent_text:
                        intents.append(intent_text)
            
            return {
                'num_intents': len(intents) if intents else 1,
                'intents': intents if intents else [query]
            }
            
        except Exception as e:
            print(f"❌ Query decomposition error: {e}")
            return {'num_intents': 1, 'intents': [query]}
    
    def process_call_turn(self, session_id: str, query: str) -> Dict:
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Invalid session'}
        
        language = session.get('language', 'en')
        original_query = query
        
        print(f"\n{'='*60}")
        print(f"🔄 Processing Turn (Session: {session_id[:8]}...)")
        print(f"{'='*60}")
        
        detected_language = self._detect_language(query)
        print(f"1️⃣ Language Detected: {detected_language}")
        
        if detected_language == 'ar':
            query = self._translate_to_english(query)
            print(f"2️⃣ Translated to English: {query}")
        
        query_type = self._classify_query_type(query)
        print(f"3️⃣ Query Type: {query_type}")
        
        if query_type != 'BANKING':
            print(f"💬 Soft query detected - skipping RAG")
            response = self._generate_soft_response(query, query_type, language)
            
            self.add_to_history(session_id, {
                'query': original_query,
                'response': response,
                'language': language,
                'intents': [query_type]
            })
            
            print(f"{'='*60}\n")
            
            return {
                'query': original_query,
                'response': response,
                'language': language,
                'decomposition': {'num_intents': 1, 'intents': [query_type]},
                'intent_results': []
            }
        
        print(f"🏦 Banking query - proceeding with RAG")
        decomposition = self._decompose_query(query)
        print(f"4️⃣ Intents Found: {decomposition['num_intents']}")
        for i, intent in enumerate(decomposition['intents'], 1):
            print(f"   Intent {i}: {intent}")
        
        intent_results = []
        intent_type = 'multi_intent' if decomposition['num_intents'] > 1 else 'single_intent'
        
        for intent in decomposition['intents']:
            print(f"\n5️⃣ Retrieving for: '{intent}'")
            
            retrieved_docs = self.retrieve_for_intent(intent)
            validation = self.validate_retrieval(retrieved_docs, intent_type)
            
            print(f"   ✓ Retrieved {len(retrieved_docs)} docs")
            print(f"   ✓ Can Answer: {validation['can_answer']} (Confidence: {validation['confidence']})")
            
            intent_results.append({
                'intent': intent,
                'retrieved_docs': retrieved_docs,
                'validation': validation
            })
        
        print(f"\n6️⃣ Generating Response...")
        response = self.generate_response(query, intent_results, language)
        print(f"   ✓ Response generated ({len(response)} chars)")
        
        self.add_to_history(session_id, {
            'query': original_query,
            'response': response,
            'language': language,
            'intents': decomposition['intents']
        })
        
        print(f"{'='*60}\n")
        
        return {
            'query': original_query,
            'response': response,
            'language': language,
            'decomposition': decomposition,
            'intent_results': intent_results
        }


class StrictVAD:
    def __init__(self, aggressiveness: int = 3):
        self.sample_rate = 16000
        self.energy_threshold = 800
        self.min_speech_duration = 1.5
        self.min_amplitude = 3000
    
    def has_speech(self, audio_bytes: bytes) -> bool:
        try:
            min_size = int(self.sample_rate * self.min_speech_duration * 2)
            if len(audio_bytes) < min_size:
                return False
            
            import struct
            num_samples = len(audio_bytes) // 2
            if num_samples == 0:
                return False
            
            audio_data = struct.unpack(f'{num_samples}h', audio_bytes[:num_samples*2])
            
            rms = (sum(s*s for s in audio_data) / num_samples) ** 0.5
            max_amp = max(abs(s) for s in audio_data)
            
            has_energy = rms > self.energy_threshold
            has_amplitude = max_amp > self.min_amplitude
            
            result = has_energy and has_amplitude
            print(f"🔊 VAD - RMS: {rms:.1f}, Max: {max_amp}, Speech: {result}")
            return result
            
        except Exception as e:
            print(f"❌ VAD error: {e}")
            return False



print("🚀 Initializing RAG System...")
rag_system = BilingualVoiceBankingRAG(GROQ_API_KEY)
app = FastAPI()

HTML_CONTENT = """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title> Nefercall </title>
<style>
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;display:flex;justify-content:center;align-items:center;padding:20px;margin:0}
.container{background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);max-width:900px;width:100%;padding:40px}
h1{text-align:center;color:#667eea;margin-bottom:10px;font-size:2.5em}
.subtitle{text-align:center;color:#666;margin-bottom:30px;font-size:1.1em}
.language-selector{display:flex;gap:20px;justify-content:center;margin-bottom:30px}
.language-btn{padding:15px 40px;border:2px solid #667eea;background:white;color:#667eea;border-radius:50px;font-size:1.1em;cursor:pointer;transition:all 0.3s;font-weight:600}
.language-btn.active{background:#667eea;color:white;transform:scale(1.05)}
.mic-indicator{width:120px;height:120px;border-radius:50%;margin:0 auto 20px;display:flex;align-items:center;justify-content:center;font-size:3.5em;background:#e2e8f0;transition:all 0.3s;box-shadow:0 4px 15px rgba(0,0,0,0.1)}
.mic-indicator.listening{background:#48bb78;animation:pulse 2s infinite;box-shadow:0 0 30px rgba(72,187,120,0.5)}
.mic-indicator.speaking{background:#667eea;animation:glow 1.5s infinite;box-shadow:0 0 30px rgba(102,126,234,0.5)}
.mic-indicator.closed{background:#e2e8f0}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.08)}}
@keyframes glow{0%,100%{opacity:1}50%{opacity:0.7}}
#status{text-align:center;font-weight:bold;color:#666;margin-bottom:20px;font-size:1.1em}
.controls{display:flex;gap:20px;justify-content:center;margin:30px 0}
.btn{padding:15px 50px;border:none;border-radius:50px;font-size:1.2em;cursor:pointer;font-weight:600;transition:all 0.3s;box-shadow:0 4px 15px rgba(0,0,0,0.2)}
.btn:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,0.3)}
.btn-start{background:#48bb78;color:white}
.btn-end{background:#f56565;color:white}
.btn:disabled{opacity:0.5;cursor:not-allowed;transform:none}
.status-box{background:#f7fafc;border-left:4px solid #667eea;padding:20px;margin-bottom:20px;border-radius:8px;font-size:1em}
.conversation{background:#f7fafc;border-radius:12px;padding:20px;max-height:450px;overflow-y:auto;margin-bottom:20px}
.message{margin-bottom:15px;padding:15px;border-radius:12px;animation:slideIn 0.3s ease}
@keyframes slideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.message.user{background:#e6f2ff;border-left:4px solid #3182ce}
.message.agent{background:#e6ffe6;border-left:4px solid #48bb78}
.message strong{display:block;margin-bottom:5px;font-size:0.9em;text-transform:uppercase;letter-spacing:0.5px}
</style>
</head>
<body>
<div class="container">
<h1> Nefercall </h1>
<p class="subtitle"> Customer Service chatbot </p>
<div class="language-selector">
<button class="language-btn active" id="btn-en" onclick="selectLang('en')">🇬🇧 English</button>
<button class="language-btn" id="btn-ar" onclick="selectLang('ar')">🇸🇦 العربية</button>
</div>
<div class="mic-indicator closed" id="mic">🎤</div>
<div id="status">Ready to start</div>
<div class="controls">
<button class="btn btn-start" id="start" onclick="start()">📞 Start Call</button>
<button class="btn btn-end" id="end" onclick="end()" disabled>❌ End Call</button>
</div>
<div class="conversation" id="conv"></div>
</div>
<script>
let ws=null,lang='en',recorder=null,chunks=[],audio=null,micOpen=false;
let audioContext=null,analyser=null,silenceTimer=null,isSpeaking=false;
let silenceThreshold=1500;
let minSpeechDuration=800;

function selectLang(l){
lang=l;
document.getElementById('btn-en').classList.toggle('active',l==='en');
document.getElementById('btn-ar').classList.toggle('active',l==='ar');
}

async function start(){
try{
const protocol=window.location.protocol==='https:'?'wss:':'ws:';
ws=new WebSocket(`${protocol}//${window.location.host}/ws`);

ws.onopen=()=>{
ws.send(JSON.stringify({type:'start',language:lang}));
document.getElementById('status').textContent='🔗 Connected, waiting for greeting...';
};

ws.onmessage=async(e)=>{
const d=JSON.parse(e.data);

if(d.type==='greeting'||d.type==='response'){
muteMic();
document.getElementById('mic').classList.remove('listening');
document.getElementById('mic').classList.add('speaking');
document.getElementById('status').textContent='🔊 Agent speaking...';
addMsg('agent',d.text);
await playAudio(d.audio);
document.getElementById('status').textContent='🎤 Your turn - speak now...';
openMic();
}else if(d.type==='transcription'){
addMsg('user',d.text);
}else if(d.type==='status'){
document.getElementById('info').textContent=d.message;
if(d.state==='processing'){
muteMic();
document.getElementById('mic').classList.remove('listening','speaking');
document.getElementById('mic').classList.add('closed');
document.getElementById('status').textContent='⚙️ Processing your request...';
}
}
};

ws.onerror=(err)=>{
console.error('WebSocket error:',err);
document.getElementById('status').textContent='❌ Connection error';
};

ws.onclose=()=>{
document.getElementById('status').textContent='Connection closed';
};

const stream=await navigator.mediaDevices.getUserMedia({audio:true});

audioContext=new(window.AudioContext||window.webkitAudioContext)();
const source=audioContext.createMediaStreamSource(stream);
analyser=audioContext.createAnalyser();
analyser.fftSize=2048;
source.connect(analyser);

recorder=new MediaRecorder(stream);

recorder.ondataavailable=(e)=>{
if(e.data.size>0&&micOpen){
chunks.push(e.data);
}
};

recorder.onstop=async()=>{
if(chunks.length>0&&micOpen){
const blob=new Blob(chunks,{type:'audio/wav'});
const totalSize=blob.size;
chunks=[];
if(totalSize>10000){
const arrayBuffer=await blob.arrayBuffer();
const uint8Array=new Uint8Array(arrayBuffer);
if(ws&&ws.readyState===WebSocket.OPEN){
muteMic();
document.getElementById('status').textContent='📤 Sending audio...';
ws.send(JSON.stringify({type:'audio',data:Array.from(uint8Array)}));
}
}else{
console.log('⚠️ Audio too short, ignoring');
if(micOpen){
setTimeout(()=>recorder.start(),100);
}
}
}
};

document.getElementById('start').disabled=true;
document.getElementById('end').disabled=false;
document.getElementById('status').textContent='✅ Call started';

}catch(err){
alert('❌ Error: '+err.message);
console.error(err);
}
}

function detectSilence(){
if(!analyser||!micOpen)return;

const bufferLength=analyser.fftSize;
const dataArray=new Uint8Array(bufferLength);
analyser.getByteTimeDomainData(dataArray);

let sum=0;
for(let i=0;i<bufferLength;i++){
const v=(dataArray[i]-128)/128;
sum+=v*v;
}
const rms=Math.sqrt(sum/bufferLength);
const volume=rms*100;

const volumeThreshold=2;

if(volume>volumeThreshold){
if(!isSpeaking){
isSpeaking=true;
console.log('🗣️ Speech started');
}
if(silenceTimer){
clearTimeout(silenceTimer);
silenceTimer=null;
}
}else{
if(isSpeaking&&!silenceTimer){
silenceTimer=setTimeout(()=>{
console.log('🤐 Silence detected - ending speech');
isSpeaking=false;
silenceTimer=null;
if(recorder&&recorder.state==='recording'){
recorder.stop();
}
},silenceThreshold);
}
}

if(micOpen){
requestAnimationFrame(detectSilence);
}
}

function openMic(){
if(!micOpen&&recorder){
chunks=[];
isSpeaking=false;
if(silenceTimer){
clearTimeout(silenceTimer);
silenceTimer=null;
}
micOpen=true;

if(recorder.state==='inactive'){
recorder.start();
}
document.getElementById('mic').classList.remove('closed','speaking');
document.getElementById('mic').classList.add('listening');
console.log('🎤 Mic opened - monitoring for speech');

detectSilence();
}
}

function muteMic(){
if(silenceTimer){
clearTimeout(silenceTimer);
silenceTimer=null;
}
isSpeaking=false;
micOpen=false;
if(recorder){
if(recorder.state==='recording'){
recorder.stop();
}
}
chunks=[];
document.getElementById('mic').classList.remove('listening','speaking');
document.getElementById('mic').classList.add('closed');
console.log('🔇 Mic muted');
}

function end(){
muteMic();
if(ws){
ws.send(JSON.stringify({type:'end'}));
ws.close();
}
if(recorder&&recorder.stream){
recorder.stream.getTracks().forEach(track=>track.stop());
}
if(audio){
audio.pause();
audio=null;
}
if(audioContext){
audioContext.close();
audioContext=null;
}
document.getElementById('start').disabled=false;
document.getElementById('end').disabled=true;
document.getElementById('mic').classList.remove('listening','speaking');
document.getElementById('mic').classList.add('closed');
document.getElementById('status').textContent='Call ended';
}

function addMsg(type,text){
const conv=document.getElementById('conv');
const div=document.createElement('div');
div.className='message '+type;
div.innerHTML=`<strong>${type==='user'?'👤 You':'🤖 Agent'}:</strong> ${text}`;
conv.appendChild(div);
conv.scrollTop=conv.scrollHeight;
}

async function playAudio(base64Audio){
if(audio){
audio.pause();
audio=null;
}
return new Promise((resolve)=>{
audio=new Audio('data:audio/mp3;base64,'+base64Audio);
audio.onended=()=>{
document.getElementById('mic').classList.remove('speaking');
console.log('🔊 Agent finished speaking');
resolve();
};
audio.onerror=(err)=>{
console.error('Audio playback error:',err);
resolve();
};
audio.play().catch(e=>{
console.error('Audio play failed:',e);
resolve();
});
});
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return HTML_CONTENT

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    vad = StrictVAD(aggressiveness=3)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'start':
                language = message.get('language', 'en')
                session_id = rag_system.create_session(initial_language=language)
                
                greeting = rag_system._generate_dynamic_greeting(language)
                greeting_audio = rag_system.text_to_speech(greeting, language=language)
                greeting_audio_b64 = __import__('base64').b64encode(greeting_audio).decode('utf-8')
                
                await websocket.send_json({
                    'type': 'greeting',
                    'text': greeting,
                    'audio': greeting_audio_b64
                })
            
            elif message['type'] == 'audio' and session_id:
                audio_data = bytes(message['data'])
                
                if vad.has_speech(audio_data):
                    await websocket.send_json({
                        'type': 'status',
                        'message': 'Processing your request...',
                        'state': 'processing'
                    })
                    
                    session = rag_system.get_session(session_id)
                    language = session.get('language', 'en')
                    
                    text = rag_system.transcribe_audio(audio_data, language)
                    
                    if text:
                        await websocket.send_json({
                            'type': 'transcription',
                            'text': text
                        })
                        
                        result = rag_system.process_call_turn(session_id, text)
                        
                        response_audio = rag_system.text_to_speech(
                            result['response'],
                            language=result.get('language', 'en')
                        )
                        response_audio_b64 = __import__('base64').b64encode(response_audio).decode('utf-8')
                        
                        await websocket.send_json({
                            'type': 'response',
                            'text': result['response'],
                            'audio': response_audio_b64
                        })
                else:
                    print("⚠️ No speech detected, ignoring audio chunk")
                    session = rag_system.get_session(session_id)
                    language = session.get('language', 'en')
                    if language == 'ar':
                        no_speech_msg = "عذراً، لم أتمكن من سماعك بوضوح. هل يمكنك إعادة سؤالك من فضلك؟"
                    else:
                        no_speech_msg = "Sorry, I couldn't hear you clearly. Could you please repeat your question?"
                    no_speech_audio = rag_system.text_to_speech(no_speech_msg, language=language)
                    no_speech_audio_b64 = __import__('base64').b64encode(no_speech_audio).decode('utf-8')
        
                    await websocket.send_json({
                   'type': 'response', 
                   'text': no_speech_msg,
                   'audio': no_speech_audio_b64
                    })

                        
            elif message['type'] == 'end':
                if session_id:
                    summary = rag_system.end_session(session_id)
                    await websocket.send_json({
                        'type': 'status',
                        'message': f"Call ended. {summary.get('total_turns', 0)} turns completed."
                    })
                break
    
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected for session {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 STARTING BILINGUAL VOICE BANKING RAG SERVER")
    print("="*80)
    print(f"\n📍 Server URL: http://{HOST}:{PORT}")
    print("="*80 + "\n")
    
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info",
            access_log=False
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped. Goodbye!")
        import sys
        sys.exit(0)
        
