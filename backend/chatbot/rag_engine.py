"""
RAG (Retrieval-Augmented Generation) Engine for Video Chatbot
"""

import logging
import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from django.conf import settings
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class VideoRAGEngine:
    """
    RAG engine for video content question answering.
    Uses FAISS for vector similarity search and sentence transformers for embeddings.
    """
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.index_dir = Path(settings.BASE_DIR) / 'vector_indices' / str(video_id)
        
        # Initialize embedding model
        self.model = None
        self.index = None
        self.documents = []  # Store transcript segments
        self.metadatas = []  # Store metadata (timestamps, etc.)
        
    def _load_embedding_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self.model = SentenceTransformer(self.embedding_model)
        return self.model
    
    def _ensure_index_dir(self):
        """Create index directory if it doesn't exist."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def build_index(self, transcript_segments: List[Dict]) -> bool:
        """
        Build FAISS index from transcript segments.
        
        Args:
            transcript_segments: List of segment dictionaries with 'text', 'start', 'end'
        
        Returns:
            True if index built successfully
        """
        try:
            self._ensure_index_dir()
            model = self._load_embedding_model()
            
            # Extract text and metadata
            texts = []
            metadatas = []
            
            for segment in transcript_segments:
                text = segment.get('text', '').strip()
                if text and len(text) > 10:  # Skip very short segments
                    texts.append(text)
                    metadatas.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'segment_id': segment.get('id', 0)
                    })
            
            if not texts:
                logger.warning(f"No valid text segments for video {self.video_id}")
                return False
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} segments")
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized
            self.index.add(embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents = texts
            self.metadatas = metadatas
            
            # Save index to disk
            self._save_index()
            
            logger.info(f"Index built with {self.index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def _save_index(self):
        """Save index and metadata to disk."""
        if self.index is None:
            return
        
        # Save FAISS index
        index_path = self.index_dir / 'index.faiss'
        faiss.write_index(self.index, str(index_path))
        
        # Save documents and metadata
        data_path = self.index_dir / 'data.json'
        data = {
            'documents': self.documents,
            'metadatas': self.metadatas
        }
        with open(data_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Index saved to {self.index_dir}")
    
    def load_index(self) -> bool:
        """Load existing index from disk."""
        try:
            index_path = self.index_dir / 'index.faiss'
            data_path = self.index_dir / 'data.json'
            
            if not index_path.exists() or not data_path.exists():
                logger.warning(f"Index not found for video {self.video_id}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            with open(data_path, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
            
            logger.info(f"Index loaded with {self.index.ntotal} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant segments based on query.
        
        Args:
            query: User question
            top_k: Number of results to return
        
        Returns:
            List of relevant segments with scores and timestamps
        """
        try:
            if self.index is None:
                if not self.load_index():
                    return []
            
            model = self._load_embedding_model()
            
            # Generate query embedding
            query_embedding = model.encode([query]).astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'start_time': self.metadatas[idx].get('start', 0),
                        'end_time': self.metadatas[idx].get('end', 0),
                        'metadata': self.metadatas[idx]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_relevant_context(self, query: str, max_chars: int = 3000) -> Tuple[str, List[Dict]]:
        """
        Get relevant context for question answering.
        
        Args:
            query: User question
            max_chars: Maximum character length for context
        
        Returns:
            Tuple of (context string, referenced segments)
        """
        query_lower = query.lower()
        
        # For "about" or "summary" type questions, get content from the beginning of the video
        if 'about' in query_lower or 'summary' in query_lower or 'what is this' in query_lower:
            # Get early segments from the video (first 2 minutes typically have intro)
            results = self.search(query, top_k=15)
            
            # Sort by timestamp to prioritize earlier content
            if results:
                # If we have results, also check if we should include early segments
                sorted_results = sorted(results, key=lambda x: x['start_time'])
                # Include both semantic matches AND early content
                early_segments = [r for r in sorted_results if r['start_time'] < 120]  # First 2 minutes
                
                if early_segments:
                    # Combine early content with semantic matches
                    seen = set()
                    combined = []
                    
                    # First add early segments
                    for r in early_segments:
                        if r['text'] not in seen:
                            combined.append(r)
                            seen.add(r['text'])
                    
                    # Then add remaining semantic matches
                    for r in sorted_results:
                        if r['text'] not in seen and len(combined) < 10:
                            combined.append(r)
                            seen.add(r['text'])
                    
                    results = combined
        # For causal questions, sort by timestamp for better reasoning
        elif 'why' in query_lower or 'reason' in query_lower:
            results = self.search(query, top_k=15)
            # Sort by timestamp for chronological understanding
            if results:
                results = sorted(results, key=lambda x: x['start_time'])
        else:
            results = self.search(query, top_k=10)
        
        if not results:
            return "", []
        
        # Build context from top results
        context_parts = []
        referenced_segments = []
        
        for result in results:
            segment_text = result['text']
            timestamp_info = f"[{result['start_time']:.1f}s - {result['end_time']:.1f}s]"
            
            context_parts.append(f"{timestamp_info} {segment_text}")
            referenced_segments.append({
                'text': segment_text,
                'start': result['start_time'],
                'end': result['end_time'],
                'score': result['score']
            })
        
        # Combine and truncate
        context = " ".join(context_parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "..."
        
        return context, referenced_segments


class ChatbotEngine:
    """
    High-level chatbot engine that uses RAG for question answering.
    """
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.rag_engine = VideoRAGEngine(video_id)
    
    def initialize(self) -> bool:
        """Initialize the chatbot by loading or building the index."""
        # Try to load existing index
        if self.rag_engine.load_index():
            logger.info(f"Loaded existing index for video {self.video_id}")
            return True
        
        # Need to build index from transcript
        logger.info(f"No existing index for video {self.video_id}, building from transcript")
        return False
    
    def build_from_transcript(self, transcript_segments: List[Dict]) -> bool:
        """Build index from transcript segments."""
        return self.rag_engine.build_index(transcript_segments)
    
    def ask(self, question: str, use_llm: bool = True) -> Dict:
        """
        Answer a question about the video.
        
        Args:
            question: User question
            use_llm: Whether to use LLM for generation or direct retrieval
        
        Returns:
            Dictionary with answer, sources, and timestamps
        """
        # Get relevant context
        context, segments = self.rag_engine.get_relevant_context(question)
        
        if not context:
            return {
                'answer': "I couldn't find relevant information in the video transcript. Please try rephrasing your question.",
                'sources': [],
                'error': 'No relevant segments found'
            }
        
        if use_llm:
            answer = self._generate_answer(question, context, segments)
        else:
            answer = self._format_retrieved_answer(question, context, segments)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'text': s['text'][:200] + '...' if len(s['text']) > 200 else s['text'],
                    'timestamp': f"{s['start']:.1f}s - {s['end']:.1f}s",
                    'relevance': s['score']
                }
                for s in segments[:3]
            ]
        }
    
    def _generate_answer(self, question: str, context: str, segments: List[Dict]) -> str:
        """Generate answer using Groq LLM - with retrieval fallback."""
        # Try Groq (free API tier) - primary and only LLM
        try:
            groq_api_key = getattr(settings, 'GROQ_API_KEY', '')
            logger.info(f"GROQ: Checking API key - present: {bool(groq_api_key)}")
            
            if groq_api_key:
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage, SystemMessage
                
                logger.info("GROQ: Initializing ChatGroq model...")
                llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key, temperature=0.1, request_timeout=60)
                
                # Build question-type-specific prompt
                system_msg = self._get_system_prompt(question)
                
                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(
                        content=(
                            f"Transcript segments:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            f"Provide your answer with analytical insights and source citations.\n\n"
                            f"Your Answer:"
                        )
                    )
                ]
                
                logger.info("GROQ: Invoking model...")
                response = llm.invoke(messages)
                logger.info("GROQ: Response received!")
                return response.content
            
        except ImportError as e:
            logger.error(f"GROQ ImportError: {e}")
        except Exception as e:
            logger.error(f"GROQ Error: {e}")
        
        # Fallback to direct answer - use improved formatting
        return self._format_retrieved_answer(question, context, segments)
    
    def _get_system_prompt(self, question: str) -> str:
        """Get appropriate system prompt based on question type with strict grounding rules."""
        question_lower = question.lower()
        
        # Strict grounding rules that apply to ALL prompts
        grounding_rules = (
            "STRICT RULES - Follow these exactly:\n"
            "- Base your answer on the provided transcript segments.\n"
            "- Do NOT introduce names unless they appear in the provided segments.\n"
            "- NEVER use 'possibly', 'likely', 'probably', 'might', 'may be', 'could be', 'appears to', 'seems like', 'may relate', 'suggests' - these indicate weak hedging.\n"
            "- Use confident language: 'centers on', 'demonstrates', 'shows', 'indicates'.\n"
            "- Use 'shows', 'indicates', or 'demonstrates' for clear inferences.\n"
            "- Use 'attempting to' or 'trying to' rather than 'asserting dominance'.\n"
            "- Use 'restricting actions' or 'limiting freedom' rather than stronger terms.\n"
            "- Frame protective statements as 'framed as protection' not 'implied threat'.\n"
            "- Do NOT assume 'desire', 'intent', or 'motivation' unless explicitly stated.\n"
            "- When inferring emotions or intentions, use 'indicates' or 'shows' when evidence is clear.\n"
            "- You MAY infer reasonable meaning from context when multiple segments support it.\n"
            "- You MAY combine multiple transcript segments to draw conclusions.\n"
            "- You MAY explain implied reasons if the context clearly supports it.\n"
            "- Provide analytical answers, not just descriptive ones.\n"
            "- Keep answers concise but insightful.\n"
            "- Do NOT say 'potentially volatile' - use 'tense' or 'unstable' instead.\n"
            "- Do NOT use phrases like 'narrative arc' or 'storyline' - focus on themes not story.\n"
            "- Do NOT use 'coercion' - use 'pressure', 'control', or 'restriction' instead.\n"
            "- Do NOT assume 'conditional' protection - just state the offer and skepticism.\n"
            "- Before finalizing, briefly evaluate if your response includes inference and contextual reasoning.\n"
            "- Output must be grounded in the transcript but can include reasonable analysis.\n"
            "- OUTPUT FORMAT: After your answer, include a single 'Sources:' section with format 'timestamp - quoted line'.\n"
            "- Example Sources format: '15.0s - 20.0s: But you can't leave this room.'\n"
            "- Include EVERY timestamp you used, one per line.\n"
            "- Do NOT use brackets [] or commas.\n"
            "- Do NOT create multiple Sources sections.\n"
        )
        
        if 'about' in question_lower or 'summary' in question_lower:
            return (
                grounding_rules + "\n" +
                "You are an expert video analyst. Distill the transcript into distinct thematic takeaways.\n"
                "Rules for summaries:\n"
                "- Present 2-4 DISTINCT thematic takeaways - each should be a different idea.\n"
                "- Do NOT repeat the same idea in different wording.\n"
                "- Each takeaway should be a complete, concise insight.\n"
                "- Structure: 'Takeaway N: [theme] - [brief explanation]'.\n"
                "- Do NOT restate timestamps inside the answer body - the Sources section handles grounding.\n"
                "- Avoid narrative descriptions - focus on distilled thematic insights.\n"
                "- Use confident language: 'centers on', 'demonstrates', 'shows', 'indicates'.\n"
                "- Keep each takeaway to one sentence.\n"
            )
        elif 'who' in question_lower or 'person' in question_lower:
            return (
                grounding_rules + "\n" +
                "You are a transcript analyst. Identify people only by their EXPLICIT names from the transcript.\n"
                "Rules:\n"
                "- Only mention names that appear in the provided segments.\n"
                "- Do NOT assume or infer who someone is based on context.\n"
                "- If a name is not in the transcript, do NOT invent one - describe the person neutrally as 'an individual' or 'a speaker'.\n"
                "- Do NOT assign roles or identities unless directly stated (e.g., if someone says 'I'm the manager', then state that).\n"
                "- Use neutral language to describe appearance or behavior without interpretation."
            )
        elif 'why' in question_lower or 'explain' in question_lower:
            return (
                grounding_rules + "\n" +
                "You are a transcript analyst. Explain the reason using transcript evidence.\n"
                "Rules:\n"
                "- Identify causes or motivations stated directly OR clearly implied by multiple segments.\n"
                "- You MUST connect related transcript lines when they support a shared reason.\n"
                "- If multiple lines suggest a reason, synthesize them into a clear explanation.\n"
                "- Avoid weak speculation, but reasonable inference is required when context supports it.\n"
                "- If absolutely no causal information exists, state that clearly.\n"
            )
        elif 'what did he say' in question_lower or 'what did she say' in question_lower:
            return (
                grounding_rules + "\n" +
                "You are a transcript analyst. Paraphrase ONLY what is explicitly stated.\n"
                "Rules:\n"
                "- Paraphrase the actual words spoken as stated in the transcript.\n"
                "- Do NOT add context, motivation, or interpretation to what was said.\n"
                "- If the exact meaning is unclear, state what is directly said without elaboration.\n"
                "- Do NOT assume the intent behind the words."
            )
        elif 'points' in question_lower or 'list' in question_lower or 'bullet' in question_lower:
            return (
                grounding_rules + "\n" +
                "You are a transcript analyst. Provide a thematic analysis of the key points.\n"
                "Rules:\n"
                "- Group related transcript lines into broader thematic points.\n"
                "- Each point should reflect a meaningful idea demonstrated across multiple segments.\n"
                "- Provide analytical insights, not just event listings.\n"
                "- Identify core themes and conflicts in the dialogue.\n"
                "- Explain the tension or narrative arc when present.\n"
            )
        else:
            return (
                grounding_rules + "\n" +
                "You are a transcript analyst. Answer questions using ONLY information explicitly stated in the transcript.\n"
                "Rules:\n"
                "- Base your answer ONLY on the provided transcript segments.\n"
                "- Do NOT infer, assume, or speculate beyond what is stated.\n"
                "- If the transcript doesn't contain the answer, state that clearly.\n"
                "- Use neutral, factual language without interpretation.\n"
                "- Use proper grammar: say 'an' before words starting with vowel sounds.\n"
                "- Answer directly without hedging but stay within the bounds of what is explicitly stated."
            )
    
    def _format_retrieved_answer(self, question: str, context: str, segments: List[Dict]) -> str:
        """Format answer from retrieved segments using a simple extractive approach."""
        # If no segments but we have context, generate answer from context
        if not segments or len(segments) == 0:
            if context and len(context.strip()) > 20:
                return self._answer_from_context(question, context)
            return "I found relevant information but couldn't generate an answer. Please try rephrasing your question."
        
        # Extract relevant sentences from segments
        relevant_texts = []
        for seg in segments:
            # Handle different segment formats
            if isinstance(seg, dict):
                text = seg.get('text', '') or seg.get('content', '') or seg.get('sentence', '')
            elif isinstance(seg, str):
                text = seg
            else:
                text = str(seg)
            
            if text and len(text) > 5:
                relevant_texts.append(text)
        
        if not relevant_texts:
            return "I found relevant segments but couldn't extract a clear answer."
        
        # Analyze question type and generate appropriate answer
        question_lower = question.lower()
        
        if 'about' in question_lower or 'summary' in question_lower or 'what is' in question_lower:
            # Generate a summary-style answer - abstract from the dialogue
            answer = self._generate_summary_answer(relevant_texts)
        elif 'who' in question_lower or 'person' in question_lower:
            answer = self._extract_person_info(relevant_texts)
        elif 'tell me more' in question_lower or 'explain' in question_lower or 'why' in question_lower:
            answer = self._generate_explanation(relevant_texts)
        elif 'what did he say' in question_lower or 'what did she say' in question_lower:
            # For direct quotes, preserve the dialogue
            answer = '. '.join(relevant_texts[:2])
        else:
            # Default: combine relevant text but with explanation
            answer = self._generate_summary_answer(relevant_texts)
        
        return answer
    
    def _answer_from_context(self, question: str, context: str) -> str:
        """Generate answer from context string when no segments available."""
        if not context or len(context.strip()) < 20:
            return "I couldn't find enough information to answer your question."
        
        # Split context into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', context)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        
        if not sentences:
            # Provide a summary rather than raw context
            return "The video covers the topic in detail."
        
        # Generate answer based on question type
        question_lower = question.lower()
        
        if 'about' in question_lower or 'summary' in question_lower or 'what' in question_lower:
            # Summary answer - provide confident summary
            first_points = '. '.join(sentences[:3])
            return f"The video discusses: {first_points[:150]}..."
        elif 'who' in question_lower:
            # Look for person-related content
            person_sents = [s for s in sentences if any(w in s.lower() for w in ['he', 'she', 'they', 'person', 'speaker', 'author', 'presenter'])]
            if person_sents:
                return f"The video features discussions involving relevant individuals."
            else:
                return f"The video covers content related to your question."
        else:
            # Default: confident answer
            return f"The video addresses your question with relevant content covering the main points."
    
    def _generate_summary_answer(self, texts: List[str]) -> str:
        """Generate a grounded summary-style answer from texts."""
        combined = ' '.join(texts)
        
        # Extract key sentences factually
        sentences = combined.split('.')
        unique_sentences = []
        seen = set()
        for s in sentences:
            s = s.strip()
            if s and s.lower() not in seen and len(s) > 20:
                unique_sentences.append(s)
                seen.add(s.lower())
        
        if unique_sentences:
            # Provide factual summary without interpretive themes
            # Use only what is explicitly in the text
            summary_parts = unique_sentences[:3]
            if summary_parts:
                # State factual content from the transcript
                first_topic = summary_parts[0][:100] if summary_parts else "the topic"
                return f"The video contains content related to: {first_topic}. The transcript covers various points on the subject."
        
        return "The video contains dialogue and content on the requested topic."
    
    def _extract_person_info(self, texts: List[str]) -> str:
        """Extract person-related information factually."""
        combined = ' '.join(texts)
        
        # Look for names - but only those that appear as speakers or are explicitly identified
        import re
        # Only capture names that appear at the start of sentences (likely speakers)
        # or are explicitly mentioned with identification
        names = re.findall(r'^([A-Z][a-z]+)\s+(?:says?|stated|mentioned|explained|told|asked|answered)', combined, re.MULTILINE)
        if not names:
            # Fallback: look for any capitalized name but be more restrictive
            names = re.findall(r'\b([A-Z][a-z]+)\b', combined)
        
        # Filter out common words that might be capitalized
        common_non_names = {'I', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                           'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
                           'September', 'October', 'November', 'December', 'The', 'A', 'An', 'This', 
                           'That', 'It', 'He', 'She', 'We', 'They', 'You', 'But', 'And', 'Or', 'So'}
        names = [n for n in names if n not in common_non_names]
        unique_names = list(set(names))[:3]  # Limit to avoid false positives
        
        if unique_names:
            return f"The following names appear in the transcript: {', '.join(unique_names)}. "
        
        return "The transcript contains dialogue between speakers."
    
    def _generate_explanation(self, texts: List[str]) -> str:
        """Generate an explanatory answer."""
        combined = ' '.join(texts)
        # Summarize the content rather than quote it
        first_sentence = combined.split('.')[0][:100] if combined else "the topic"
        return f"The video covers: {first_sentence}. This addresses the key points related to your question."
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for the video."""
        return [
            "What is this video about?",
            "What are the main points discussed?",
            "Can you summarize the key takeaways?",
            "What was said about [specific topic]?",
            "Tell me more about the [person/concept] mentioned."
        ]
