"""
Quality Assessment Framework for NeuroDoc
Implements comprehensive quality assessment for responses, citations,
and overall system performance with multiple evaluation dimensions.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import statistics
from datetime import datetime
import numpy as np

from ..utils.performance import global_performance_monitor, AsyncCache, timed_operation
from ..config import Config

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    COHERENCE = "coherence"
    DEPTH = "depth"
    OBJECTIVITY = "objectivity"
    CURRENCY = "currency"
    CREDIBILITY = "credibility"
    USEFULNESS = "usefulness"


class AssessmentLevel(Enum):
    """Levels of quality assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class ContentType(Enum):
    """Types of content that can be assessed."""
    RESPONSE = "response"
    CITATION = "citation"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"


@dataclass
class QualityMetric:
    """A single quality metric measurement."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    explanation: str
    evidence: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAssessment:
    """Complete quality assessment for content."""
    content_id: str
    content_type: ContentType
    metrics: List[QualityMetric]
    overall_score: float
    overall_level: AssessmentLevel
    assessment_summary: str
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparativeAssessment:
    """Comparative assessment between multiple pieces of content."""
    assessments: List[QualityAssessment]
    ranking: List[str]  # Content IDs ranked by quality
    comparison_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    consensus_metrics: Dict[QualityDimension, float] = field(default_factory=dict)
    variance_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityTrend:
    """Quality trend analysis over time."""
    dimension: QualityDimension
    time_series: List[Tuple[str, float]]  # (timestamp, score)
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float
    change_points: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)


class QualityAssessor:
    """
    Comprehensive quality assessment framework for evaluating content
    across multiple dimensions with automated scoring and feedback.
    """
    
    def __init__(self, config: Config):
        """Initialize the quality assessor."""
        self.config = config
        self.dimension_assessors = self._initialize_dimension_assessors()
        self.quality_thresholds = self._initialize_quality_thresholds()
        self.assessment_cache = {}
        self.quality_history = []
        
    def _initialize_dimension_assessors(self) -> Dict[QualityDimension, callable]:
        """Initialize assessors for each quality dimension."""
        return {
            QualityDimension.ACCURACY: self._assess_accuracy,
            QualityDimension.COMPLETENESS: self._assess_completeness,
            QualityDimension.RELEVANCE: self._assess_relevance,
            QualityDimension.CLARITY: self._assess_clarity,
            QualityDimension.COHERENCE: self._assess_coherence,
            QualityDimension.DEPTH: self._assess_depth,
            QualityDimension.OBJECTIVITY: self._assess_objectivity,
            QualityDimension.CURRENCY: self._assess_currency,
            QualityDimension.CREDIBILITY: self._assess_credibility,
            QualityDimension.USEFULNESS: self._assess_usefulness
        }
    
    def _initialize_quality_thresholds(self) -> Dict[AssessmentLevel, Tuple[float, float]]:
        """Initialize quality score thresholds for different levels."""
        return {
            AssessmentLevel.EXCELLENT: (0.9, 1.0),
            AssessmentLevel.GOOD: (0.7, 0.9),
            AssessmentLevel.SATISFACTORY: (0.5, 0.7),
            AssessmentLevel.POOR: (0.3, 0.5),
            AssessmentLevel.UNACCEPTABLE: (0.0, 0.3)
        }
    
    @timed_operation("quality_assessment", global_performance_monitor)
    async def assess_content(
        self,
        content: str,
        content_type: ContentType,
        context: Optional[Dict[str, Any]] = None,
        dimensions: Optional[List[QualityDimension]] = None
    ) -> QualityAssessment:
        """
        Assess the quality of content across multiple dimensions.
        
        Args:
            content: Content to assess
            content_type: Type of content being assessed
            context: Additional context for assessment
            dimensions: Specific dimensions to assess (all if None)
            
        Returns:
            Comprehensive quality assessment
        """
        try:
            # Default to all dimensions if none specified
            if dimensions is None:
                dimensions = list(QualityDimension)
            
            # Prepare assessment context
            assessment_context = self._prepare_assessment_context(content, content_type, context)
            
            # Assess each dimension
            metrics = []
            for dimension in dimensions:
                assessor = self.dimension_assessors.get(dimension)
                if assessor:
                    metric = await assessor(content, assessment_context)
                    metrics.append(metric)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            
            # Determine quality level
            overall_level = self._determine_quality_level(overall_score)
            
            # Generate assessment summary
            assessment_summary = self._generate_assessment_summary(metrics, overall_level)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(metrics)
            
            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(metrics, weaknesses)
            
            # Create assessment
            assessment = QualityAssessment(
                content_id=self._generate_content_id(content),
                content_type=content_type,
                metrics=metrics,
                overall_score=overall_score,
                overall_level=overall_level,
                assessment_summary=assessment_summary,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=improvement_suggestions,
                metadata={
                    "content_length": len(content),
                    "assessed_dimensions": [d.value for d in dimensions],
                    "context_provided": context is not None
                }
            )
            
            # Store in history for trend analysis
            self.quality_history.append(assessment)
            
            logger.info(f"Assessed {content_type.value} content: {overall_level.value} ({overall_score:.2f})")
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing content quality: {e}")
            raise
    
    def _prepare_assessment_context(
        self,
        content: str,
        content_type: ContentType,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare context for quality assessment."""
        assessment_context = {
            "content": content,
            "content_type": content_type,
            "content_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(re.split(r'[.!?]+', content)),
            "paragraph_count": len(content.split('\n\n')),
        }
        
        if context:
            assessment_context.update(context)
        
        return assessment_context
    
    def _generate_content_id(self, content: str) -> str:
        """Generate a unique ID for content."""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    async def _assess_accuracy(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the accuracy of content."""
        score = 0.7  # Base score
        confidence = 0.8
        evidence = []
        suggestions = []
        
        # Check for factual indicators
        factual_indicators = [
            "research shows", "studies indicate", "data reveals",
            "according to", "evidence suggests", "findings demonstrate"
        ]
        
        factual_statements = sum(1 for indicator in factual_indicators if indicator in content.lower())
        if factual_statements > 0:
            score += min(0.2, factual_statements * 0.05)
            evidence.append(f"Found {factual_statements} factual indicators")
        
        # Check for citation markers
        citation_patterns = [r'\[?\d+\]?', r'\([^)]+,\s*\d{4}\)', r'\(\w+\s+et\s+al\.?,?\s*\d{4}\)']
        citations = 0
        for pattern in citation_patterns:
            citations += len(re.findall(pattern, content))
        
        if citations > 0:
            score += min(0.15, citations * 0.03)
            evidence.append(f"Found {citations} citation markers")
        else:
            suggestions.append("Add citations to support factual claims")
        
        # Check for hedging language (indicates uncertainty)
        hedging_terms = ["may", "might", "possibly", "potentially", "suggests", "indicates"]
        hedging_count = sum(1 for term in hedging_terms if term in content.lower().split())
        
        if hedging_count > len(content.split()) * 0.1:  # Too much hedging
            score -= 0.1
            suggestions.append("Reduce excessive hedging language for more confident statements")
        
        # Check for absolute statements without evidence
        absolute_terms = ["always", "never", "all", "none", "definitely", "certainly"]
        absolute_count = sum(1 for term in absolute_terms if term in content.lower().split())
        
        if absolute_count > 0 and citations == 0:
            score -= min(0.2, absolute_count * 0.05)
            suggestions.append("Provide evidence for absolute statements or use more nuanced language")
        
        explanation = f"Accuracy assessment based on factual indicators, citations, and language patterns"
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_completeness(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the completeness of content."""
        score = 0.5  # Base score
        confidence = 0.9
        evidence = []
        suggestions = []
        
        # Check content length adequacy
        word_count = context.get("word_count", 0)
        content_type = context.get("content_type")
        
        # Expected lengths by content type
        expected_lengths = {
            ContentType.RESPONSE: (100, 500),
            ContentType.SUMMARY: (50, 200),
            ContentType.EXPLANATION: (150, 600),
            ContentType.ANALYSIS: (200, 800),
            ContentType.COMPARISON: (150, 500)
        }
        
        if content_type in expected_lengths:
            min_length, ideal_length = expected_lengths[content_type]
            
            if word_count >= ideal_length:
                score += 0.3
                evidence.append(f"Content length ({word_count} words) meets ideal length")
            elif word_count >= min_length:
                score += 0.2
                evidence.append(f"Content length ({word_count} words) meets minimum requirements")
            else:
                score -= 0.2
                suggestions.append(f"Expand content (current: {word_count} words, minimum: {min_length})")
        
        # Check for structural completeness
        has_introduction = any(
            indicator in content.lower()
            for indicator in ["introduction", "overview", "begin", "first", "initially"]
        )
        
        has_conclusion = any(
            indicator in content.lower()
            for indicator in ["conclusion", "summary", "finally", "in conclusion", "to conclude"]
        )
        
        if has_introduction and has_conclusion:
            score += 0.2
            evidence.append("Contains both introduction and conclusion")
        elif has_introduction or has_conclusion:
            score += 0.1
            if not has_introduction:
                suggestions.append("Add an introduction to provide context")
            if not has_conclusion:
                suggestions.append("Add a conclusion to summarize key points")
        else:
            suggestions.append("Add clear introduction and conclusion")
        
        # Check for comprehensive coverage
        query = context.get("query", "")
        if query:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            coverage = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            
            if coverage > 0.8:
                score += 0.2
                evidence.append(f"High query coverage ({coverage:.1%})")
            elif coverage > 0.5:
                score += 0.1
                evidence.append(f"Moderate query coverage ({coverage:.1%})")
            else:
                suggestions.append("Address more aspects of the original query")
        
        explanation = f"Completeness based on length, structure, and query coverage"
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_relevance(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the relevance of content to the query."""
        score = 0.5  # Base score
        confidence = 0.85
        evidence = []
        suggestions = []
        
        query = context.get("query", "")
        if not query:
            confidence = 0.3
            return QualityMetric(
                dimension=QualityDimension.RELEVANCE,
                score=score,
                confidence=confidence,
                explanation="Cannot assess relevance without query context",
                evidence=evidence,
                suggestions=["Provide query context for relevance assessment"]
            )
        
        # Keyword overlap analysis
        query_words = set(word.lower() for word in query.split() if len(word) > 2)
        content_words = set(word.lower() for word in content.split() if len(word) > 2)
        
        if query_words:
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            score += keyword_overlap * 0.3
            evidence.append(f"Keyword overlap: {keyword_overlap:.1%}")
        
        # Semantic relevance indicators
        content_lower = content.lower()
        
        # Check if content directly addresses the query
        direct_addressing = any(
            phrase in content_lower
            for phrase in ["in response to", "regarding", "about", "concerning"]
        )
        
        if direct_addressing:
            score += 0.1
            evidence.append("Content directly addresses the query")
        
        # Check for off-topic content
        off_topic_indicators = [
            "unrelated", "different topic", "another subject", "changing subject"
        ]
        
        if any(indicator in content_lower for indicator in off_topic_indicators):
            score -= 0.2
            suggestions.append("Focus more closely on the original query")
        
        # Check query intent fulfillment
        query_lower = query.lower()
        intent_fulfillment = 0
        
        if "what" in query_lower and any(def_word in content_lower for def_word in ["definition", "is", "means"]):
            intent_fulfillment += 0.2
        
        if "how" in query_lower and any(proc_word in content_lower for proc_word in ["process", "steps", "method"]):
            intent_fulfillment += 0.2
        
        if "why" in query_lower and any(exp_word in content_lower for exp_word in ["because", "reason", "cause"]):
            intent_fulfillment += 0.2
        
        if "compare" in query_lower and any(comp_word in content_lower for comp_word in ["comparison", "versus", "differ"]):
            intent_fulfillment += 0.2
        
        score += intent_fulfillment
        if intent_fulfillment > 0:
            evidence.append(f"Fulfills query intent (score: {intent_fulfillment:.1f})")
        else:
            suggestions.append("Better address the specific intent of the query")
        
        explanation = f"Relevance based on keyword overlap and query intent fulfillment"
        
        return QualityMetric(
            dimension=QualityDimension.RELEVANCE,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_clarity(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the clarity and readability of content."""
        score = 0.5  # Base score
        confidence = 0.9
        evidence = []
        suggestions = []
        
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            if 10 <= avg_sentence_length <= 20:  # Optimal range
                score += 0.2
                evidence.append(f"Good average sentence length ({avg_sentence_length:.1f} words)")
            elif avg_sentence_length > 30:
                score -= 0.1
                suggestions.append("Break down long sentences for better readability")
            elif avg_sentence_length < 5:
                score -= 0.1
                suggestions.append("Combine very short sentences for better flow")
        
        # Vocabulary complexity
        complex_words = 0
        total_words = 0
        
        for word in content.split():
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if len(word_clean) > 2:
                total_words += 1
                if len(word_clean) > 8:  # Arbitrary complexity threshold
                    complex_words += 1
        
        if total_words > 0:
            complexity_ratio = complex_words / total_words
            if complexity_ratio < 0.2:  # Good balance
                score += 0.15
                evidence.append(f"Appropriate vocabulary complexity ({complexity_ratio:.1%})")
            elif complexity_ratio > 0.4:
                score -= 0.1
                suggestions.append("Use simpler vocabulary where possible")
        
        # Structural clarity
        has_bullet_points = '•' in content or re.search(r'\n\s*[-*]\s+', content)
        has_numbered_lists = re.search(r'\n\s*\d+\.\s+', content)
        has_clear_paragraphs = '\n\n' in content
        
        structure_score = 0
        if has_bullet_points or has_numbered_lists:
            structure_score += 0.1
            evidence.append("Uses lists for better organization")
        
        if has_clear_paragraphs:
            structure_score += 0.1
            evidence.append("Well-organized with clear paragraphs")
        
        if structure_score == 0:
            suggestions.append("Use lists, bullet points, or clear paragraphs to improve structure")
        
        score += structure_score
        
        # Jargon and technical terms
        jargon_indicators = [
            "utilize", "methodology", "paradigm", "facilitate", "implement",
            "instantiate", "operationalize", "conceptualize"
        ]
        
        jargon_count = sum(1 for term in jargon_indicators if term in content.lower())
        if jargon_count > 3:
            score -= 0.1
            suggestions.append("Replace jargon with simpler terms where possible")
        
        explanation = f"Clarity based on sentence structure, vocabulary, and organization"
        
        return QualityMetric(
            dimension=QualityDimension.CLARITY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_coherence(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the logical flow and coherence of content."""
        score = 0.6  # Base score
        confidence = 0.7
        evidence = []
        suggestions = []
        
        # Transition words and phrases
        transition_words = [
            "however", "therefore", "furthermore", "moreover", "additionally",
            "consequently", "meanwhile", "similarly", "in contrast", "for example",
            "specifically", "in conclusion", "finally", "first", "second", "next"
        ]
        
        transition_count = sum(1 for word in transition_words if word in content.lower())
        paragraph_count = max(1, len(content.split('\n\n')))
        
        transition_ratio = transition_count / paragraph_count
        if transition_ratio > 0.5:
            score += 0.2
            evidence.append(f"Good use of transitions ({transition_count} transitions)")
        elif transition_ratio < 0.2:
            suggestions.append("Add transition words to improve flow between ideas")
        
        # Logical progression
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Check for logical connectors
        logical_connectors = ["because", "since", "as a result", "due to", "leads to", "causes"]
        connector_count = sum(1 for connector in logical_connectors 
                            for sentence in sentences if connector in sentence.lower())
        
        if connector_count > 0:
            score += min(0.15, connector_count * 0.05)
            evidence.append(f"Uses logical connectors ({connector_count})")
        
        # Topic consistency
        # Simple check: look for repeated key terms throughout
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Focus on substantial words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find words that appear multiple times (indicating consistent theme)
        repeated_words = [word for word, freq in word_freq.items() if freq > 2]
        if len(repeated_words) > 3:
            score += 0.1
            evidence.append("Maintains consistent terminology throughout")
        
        # Check for contradictions
        contradiction_indicators = [
            ("not", "is"), ("cannot", "can"), ("never", "always"),
            ("impossible", "possible"), ("wrong", "correct")
        ]
        
        contradiction_found = False
        for neg, pos in contradiction_indicators:
            if neg in content.lower() and pos in content.lower():
                # Simple proximity check
                neg_positions = [i for i, word in enumerate(words) if word == neg]
                pos_positions = [i for i, word in enumerate(words) if word == pos]
                
                for neg_pos in neg_positions:
                    for pos_pos in pos_positions:
                        if abs(neg_pos - pos_pos) < 50:  # Within 50 words
                            contradiction_found = True
                            break
                    if contradiction_found:
                        break
        
        if contradiction_found:
            score -= 0.15
            suggestions.append("Check for potential contradictions in the content")
        
        explanation = f"Coherence based on transitions, logical flow, and consistency"
        
        return QualityMetric(
            dimension=QualityDimension.COHERENCE,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_depth(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the depth and thoroughness of content."""
        score = 0.5  # Base score
        confidence = 0.8
        evidence = []
        suggestions = []
        
        # Analysis indicators
        analysis_indicators = [
            "analysis", "examine", "investigate", "explore", "consider",
            "evaluate", "assess", "study", "research", "findings"
        ]
        
        analysis_count = sum(1 for indicator in analysis_indicators if indicator in content.lower())
        if analysis_count > 0:
            score += min(0.2, analysis_count * 0.03)
            evidence.append(f"Contains analytical language ({analysis_count} indicators)")
        
        # Evidence and examples
        example_indicators = [
            "for example", "for instance", "such as", "including",
            "specifically", "namely", "e.g.", "i.e."
        ]
        
        example_count = sum(1 for indicator in example_indicators if indicator in content.lower())
        if example_count > 0:
            score += min(0.15, example_count * 0.05)
            evidence.append(f"Provides examples ({example_count} found)")
        else:
            suggestions.append("Add specific examples to illustrate points")
        
        # Multiple perspectives
        perspective_indicators = [
            "on the other hand", "alternatively", "another view", "different perspective",
            "some argue", "others suggest", "however", "in contrast"
        ]
        
        perspective_count = sum(1 for indicator in perspective_indicators if indicator in content.lower())
        if perspective_count > 0:
            score += min(0.15, perspective_count * 0.07)
            evidence.append(f"Considers multiple perspectives ({perspective_count})")
        else:
            suggestions.append("Consider including different perspectives or viewpoints")
        
        # Technical depth
        technical_terms = 0
        specialized_vocab = 0
        
        words = content.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if len(word_clean) > 10:  # Long words often technical
                technical_terms += 1
            if word_clean.endswith(('tion', 'sion', 'ment', 'ness', 'ology', 'ism')):
                specialized_vocab += 1
        
        if technical_terms > len(words) * 0.05:  # More than 5% long words
            score += 0.1
            evidence.append(f"Uses technical terminology appropriately")
        
        # Quantitative information
        number_pattern = r'\d+(?:\.\d+)?%?'
        numbers = re.findall(number_pattern, content)
        if len(numbers) > 2:
            score += 0.1
            evidence.append(f"Includes quantitative information ({len(numbers)} numbers)")
        
        explanation = f"Depth based on analysis, examples, perspectives, and technical content"
        
        return QualityMetric(
            dimension=QualityDimension.DEPTH,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_objectivity(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the objectivity and balance of content."""
        score = 0.7  # Base score (assume neutral)
        confidence = 0.75
        evidence = []
        suggestions = []
        
        # Bias indicators
        bias_terms = [
            "obviously", "clearly", "undoubtedly", "without question",
            "everyone knows", "it's obvious", "certainly", "definitely"
        ]
        
        bias_count = sum(1 for term in bias_terms if term in content.lower())
        if bias_count > 0:
            score -= min(0.2, bias_count * 0.05)
            suggestions.append("Reduce absolute statements that may indicate bias")
        else:
            evidence.append("Avoids obviously biased language")
        
        # Emotional language
        emotional_words = [
            "terrible", "awful", "amazing", "fantastic", "horrible",
            "wonderful", "outrageous", "brilliant", "devastating", "shocking"
        ]
        
        emotional_count = sum(1 for word in emotional_words if word in content.lower())
        if emotional_count > 2:
            score -= min(0.15, emotional_count * 0.03)
            suggestions.append("Use more neutral language instead of emotional terms")
        
        # Hedging language (indicates appropriate uncertainty)
        hedging_terms = [
            "may", "might", "could", "possibly", "potentially", "suggests",
            "indicates", "appears", "seems", "likely"
        ]
        
        hedging_count = sum(1 for term in hedging_terms if term in content.lower().split())
        word_count = len(content.split())
        
        hedging_ratio = hedging_count / word_count if word_count > 0 else 0
        if 0.02 <= hedging_ratio <= 0.08:  # Appropriate level of hedging
            score += 0.1
            evidence.append("Uses appropriate hedging language")
        elif hedging_ratio > 0.1:
            suggestions.append("Reduce excessive hedging for clearer statements")
        
        # Balanced presentation
        balance_indicators = [
            ("advantages", "disadvantages"), ("benefits", "drawbacks"),
            ("pros", "cons"), ("strengths", "weaknesses"),
            ("positive", "negative"), ("support", "oppose")
        ]
        
        balance_found = False
        for pos_term, neg_term in balance_indicators:
            if pos_term in content.lower() and neg_term in content.lower():
                balance_found = True
                break
        
        if balance_found:
            score += 0.15
            evidence.append("Presents balanced perspective with multiple viewpoints")
        
        # Citation and attribution
        attribution_indicators = [
            "according to", "research shows", "studies indicate",
            "experts suggest", "data reveals"
        ]
        
        attribution_count = sum(1 for indicator in attribution_indicators if indicator in content.lower())
        if attribution_count > 0:
            score += min(0.1, attribution_count * 0.03)
            evidence.append("Attributes information to sources")
        else:
            suggestions.append("Add attribution to sources for better objectivity")
        
        explanation = f"Objectivity based on bias indicators, emotional language, and balance"
        
        return QualityMetric(
            dimension=QualityDimension.OBJECTIVITY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_currency(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the currency and timeliness of content."""
        score = 0.5  # Base score
        confidence = 0.6
        evidence = []
        suggestions = []
        
        # Look for dates and temporal references
        current_year = datetime.now().year
        
        # Find years mentioned in content
        year_pattern = r'\b(19|20)\d{2}\b'
        years = [int(year) for year in re.findall(year_pattern, content)]
        
        if years:
            most_recent_year = max(years)
            age = current_year - most_recent_year
            
            if age <= 2:
                score += 0.3
                evidence.append(f"References recent information ({most_recent_year})")
            elif age <= 5:
                score += 0.2
                evidence.append(f"References relatively recent information ({most_recent_year})")
            elif age <= 10:
                score += 0.1
                evidence.append(f"References information from {most_recent_year}")
            else:
                score -= 0.1
                suggestions.append(f"Consider updating with more recent information (latest: {most_recent_year})")
        
        # Temporal language
        recent_indicators = [
            "recent", "recently", "current", "currently", "latest",
            "new", "modern", "contemporary", "up-to-date", "fresh"
        ]
        
        outdated_indicators = [
            "old", "outdated", "obsolete", "traditional", "historical",
            "past", "former", "previous", "legacy"
        ]
        
        recent_count = sum(1 for indicator in recent_indicators if indicator in content.lower())
        outdated_count = sum(1 for indicator in outdated_indicators if indicator in content.lower())
        
        if recent_count > outdated_count:
            score += 0.1
            evidence.append("Uses language indicating current information")
        elif outdated_count > recent_count:
            score -= 0.1
            suggestions.append("Update language to reflect current state")
        
        # Technology and methodology references
        modern_tech = [
            "ai", "machine learning", "cloud", "digital", "online",
            "internet", "web", "mobile", "app", "algorithm"
        ]
        
        modern_tech_count = sum(1 for tech in modern_tech if tech in content.lower())
        if modern_tech_count > 0:
            score += min(0.1, modern_tech_count * 0.02)
            evidence.append("References modern technology and methods")
        
        explanation = f"Currency based on temporal references and modern context"
        
        return QualityMetric(
            dimension=QualityDimension.CURRENCY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_credibility(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the credibility and trustworthiness of content."""
        score = 0.6  # Base score
        confidence = 0.8
        evidence = []
        suggestions = []
        
        # Authority indicators
        authority_indicators = [
            "research", "study", "experiment", "analysis", "investigation",
            "peer-reviewed", "published", "journal", "university", "institute"
        ]
        
        authority_count = sum(1 for indicator in authority_indicators if indicator in content.lower())
        if authority_count > 0:
            score += min(0.2, authority_count * 0.03)
            evidence.append(f"References authoritative sources ({authority_count} indicators)")
        
        # Evidence-based language
        evidence_indicators = [
            "evidence shows", "data indicates", "research demonstrates",
            "studies reveal", "findings suggest", "results show"
        ]
        
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content.lower())
        if evidence_count > 0:
            score += min(0.15, evidence_count * 0.05)
            evidence.append(f"Uses evidence-based language ({evidence_count} instances)")
        else:
            suggestions.append("Support claims with evidence-based language")
        
        # Qualification and uncertainty
        qualification_terms = [
            "according to", "research suggests", "studies indicate",
            "may", "might", "appears", "seems", "likely"
        ]
        
        qualification_count = sum(1 for term in qualification_terms if term in content.lower())
        if qualification_count > 0:
            score += min(0.1, qualification_count * 0.02)
            evidence.append("Appropriately qualifies statements")
        
        # Lack of credibility indicators
        credibility_issues = [
            "i think", "i believe", "in my opinion", "personally",
            "everyone knows", "it's obvious", "common sense"
        ]
        
        issue_count = sum(1 for issue in credibility_issues if issue in content.lower())
        if issue_count > 0:
            score -= min(0.2, issue_count * 0.05)
            suggestions.append("Replace personal opinions with evidence-based statements")
        else:
            evidence.append("Avoids unsupported personal opinions")
        
        # Consistency check
        contradictory_phrases = [
            ("always", "sometimes"), ("never", "occasionally"),
            ("all", "some"), ("none", "few")
        ]
        
        contradiction_found = False
        for phrase1, phrase2 in contradictory_phrases:
            if phrase1 in content.lower() and phrase2 in content.lower():
                contradiction_found = True
                break
        
        if contradiction_found:
            score -= 0.1
            suggestions.append("Check for internal consistency in statements")
        
        explanation = f"Credibility based on authority, evidence, and consistency"
        
        return QualityMetric(
            dimension=QualityDimension.CREDIBILITY,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    async def _assess_usefulness(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> QualityMetric:
        """Assess the usefulness and practical value of content."""
        score = 0.5  # Base score
        confidence = 0.8
        evidence = []
        suggestions = []
        
        # Actionable information
        actionable_indicators = [
            "how to", "steps", "method", "process", "procedure",
            "approach", "technique", "strategy", "way to"
        ]
        
        actionable_count = sum(1 for indicator in actionable_indicators if indicator in content.lower())
        if actionable_count > 0:
            score += min(0.2, actionable_count * 0.05)
            evidence.append(f"Provides actionable information ({actionable_count} indicators)")
        
        # Practical examples
        example_indicators = [
            "example", "instance", "case", "illustration",
            "demonstration", "sample", "for instance"
        ]
        
        example_count = sum(1 for indicator in example_indicators if indicator in content.lower())
        if example_count > 0:
            score += min(0.15, example_count * 0.04)
            evidence.append(f"Includes practical examples ({example_count})")
        else:
            suggestions.append("Add practical examples to illustrate concepts")
        
        # Problem-solving content
        problem_solving = [
            "solution", "solve", "fix", "resolve", "address",
            "answer", "remedy", "approach", "handle"
        ]
        
        problem_count = sum(1 for term in problem_solving if term in content.lower())
        if problem_count > 0:
            score += min(0.15, problem_count * 0.03)
            evidence.append("Addresses problem-solving")
        
        # Specific vs. vague information
        specific_indicators = [
            r'\d+%', r'\d+\.\d+', 'specifically', 'exactly',
            'precisely', 'detailed', 'step-by-step'
        ]
        
        specific_count = 0
        for pattern in specific_indicators:
            specific_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        if specific_count > 0:
            score += min(0.1, specific_count * 0.02)
            evidence.append("Provides specific, detailed information")
        
        # Vague language penalty
        vague_terms = [
            "various", "several", "many", "some", "often",
            "usually", "generally", "typically", "frequently"
        ]
        
        vague_count = sum(1 for term in vague_terms if term in content.lower().split())
        word_count = len(content.split())
        vague_ratio = vague_count / word_count if word_count > 0 else 0
        
        if vague_ratio > 0.05:  # More than 5% vague terms
            score -= 0.1
            suggestions.append("Replace vague terms with more specific information")
        
        # Applicability indicators
        applicability_terms = [
            "can be used", "applies to", "useful for", "beneficial",
            "practical", "implementation", "application"
        ]
        
        applicability_count = sum(1 for term in applicability_terms if term in content.lower())
        if applicability_count > 0:
            score += min(0.1, applicability_count * 0.03)
            evidence.append("Discusses practical applicability")
        
        explanation = f"Usefulness based on actionability, examples, and specificity"
        
        return QualityMetric(
            dimension=QualityDimension.USEFULNESS,
            score=max(0.0, min(1.0, score)),
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _calculate_overall_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score from individual metrics."""
        if not metrics:
            return 0.0
        
        # Weighted average with confidence-based weighting
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = metric.confidence
            total_weighted_score += metric.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> AssessmentLevel:
        """Determine quality level from score."""
        for level, (min_score, max_score) in self.quality_thresholds.items():
            if min_score <= score < max_score:
                return level
        
        # Fallback
        if score >= 0.9:
            return AssessmentLevel.EXCELLENT
        elif score >= 0.7:
            return AssessmentLevel.GOOD
        elif score >= 0.5:
            return AssessmentLevel.SATISFACTORY
        elif score >= 0.3:
            return AssessmentLevel.POOR
        else:
            return AssessmentLevel.UNACCEPTABLE
    
    def _generate_assessment_summary(
        self,
        metrics: List[QualityMetric],
        level: AssessmentLevel
    ) -> str:
        """Generate a summary of the quality assessment."""
        if not metrics:
            return "No quality metrics available for assessment."
        
        avg_score = sum(m.score for m in metrics) / len(metrics)
        best_dimension = max(metrics, key=lambda m: m.score).dimension.value
        worst_dimension = min(metrics, key=lambda m: m.score).dimension.value
        
        summary = f"Overall quality: {level.value.title()} (Score: {avg_score:.2f}). "
        summary += f"Strongest in {best_dimension}, needs improvement in {worst_dimension}."
        
        return summary
    
    def _identify_strengths_weaknesses(
        self,
        metrics: List[QualityMetric]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses from metrics."""
        strengths = []
        weaknesses = []
        
        for metric in metrics:
            if metric.score >= 0.8:
                strengths.append(f"Strong {metric.dimension.value}: {metric.explanation}")
            elif metric.score <= 0.4:
                weaknesses.append(f"Weak {metric.dimension.value}: {metric.explanation}")
        
        return strengths, weaknesses
    
    def _generate_improvement_suggestions(
        self,
        metrics: List[QualityMetric],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []
        
        # Collect suggestions from individual metrics
        for metric in metrics:
            suggestions.extend(metric.suggestions)
        
        # Add general suggestions for low-scoring dimensions
        low_scoring_dimensions = [m.dimension for m in metrics if m.score <= 0.5]
        
        if QualityDimension.CLARITY in low_scoring_dimensions:
            suggestions.append("Focus on improving clarity through shorter sentences and better organization")
        
        if QualityDimension.DEPTH in low_scoring_dimensions:
            suggestions.append("Add more detailed analysis and supporting evidence")
        
        if QualityDimension.RELEVANCE in low_scoring_dimensions:
            suggestions.append("Better address the specific requirements of the query")
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10 suggestions
    
    async def compare_assessments(
        self,
        assessments: List[QualityAssessment]
    ) -> ComparativeAssessment:
        """Compare multiple quality assessments."""
        if len(assessments) < 2:
            raise ValueError("Need at least 2 assessments for comparison")
        
        # Rank assessments by overall score
        ranking = sorted(assessments, key=lambda a: a.overall_score, reverse=True)
        ranked_ids = [a.content_id for a in ranking]
        
        # Create comparison matrix
        comparison_matrix = {}
        for i, assessment1 in enumerate(assessments):
            comparison_matrix[assessment1.content_id] = {}
            for j, assessment2 in enumerate(assessments):
                if i != j:
                    comparison_matrix[assessment1.content_id][assessment2.content_id] = \
                        assessment1.overall_score - assessment2.overall_score
        
        # Calculate consensus metrics
        consensus_metrics = {}
        for dimension in QualityDimension:
            scores = []
            for assessment in assessments:
                dimension_metric = next(
                    (m for m in assessment.metrics if m.dimension == dimension),
                    None
                )
                if dimension_metric:
                    scores.append(dimension_metric.score)
            
            if scores:
                consensus_metrics[dimension] = statistics.mean(scores)
        
        # Variance analysis
        variance_analysis = {
            "score_variance": statistics.variance([a.overall_score for a in assessments]),
            "score_range": max(a.overall_score for a in assessments) - min(a.overall_score for a in assessments),
            "level_distribution": {}
        }
        
        # Level distribution
        for level in AssessmentLevel:
            count = sum(1 for a in assessments if a.overall_level == level)
            variance_analysis["level_distribution"][level.value] = count
        
        return ComparativeAssessment(
            assessments=assessments,
            ranking=ranked_ids,
            comparison_matrix=comparison_matrix,
            consensus_metrics=consensus_metrics,
            variance_analysis=variance_analysis
        )
    
    async def analyze_quality_trends(
        self,
        dimension: QualityDimension,
        time_window_hours: int = 24
    ) -> QualityTrend:
        """Analyze quality trends over time."""
        # Filter assessments within time window
        cutoff_time = datetime.now().timestamp() - (time_window_hours * 3600)
        
        relevant_assessments = []
        for assessment in self.quality_history:
            assessment_time = datetime.fromisoformat(assessment.timestamp).timestamp()
            if assessment_time >= cutoff_time:
                relevant_assessments.append(assessment)
        
        if len(relevant_assessments) < 2:
            return QualityTrend(
                dimension=dimension,
                time_series=[],
                trend_direction="insufficient_data",
                trend_strength=0.0
            )
        
        # Extract time series for the dimension
        time_series = []
        for assessment in relevant_assessments:
            dimension_metric = next(
                (m for m in assessment.metrics if m.dimension == dimension),
                None
            )
            if dimension_metric:
                time_series.append((assessment.timestamp, dimension_metric.score))
        
        if len(time_series) < 2:
            return QualityTrend(
                dimension=dimension,
                time_series=time_series,
                trend_direction="insufficient_data",
                trend_strength=0.0
            )
        
        # Analyze trend
        scores = [score for _, score in time_series]
        
        # Simple linear trend analysis
        x = list(range(len(scores)))
        y = scores
        
        if len(x) > 1:
            # Calculate correlation coefficient as trend strength
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            x_var = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
            y_var = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
            
            if x_var > 0 and y_var > 0:
                correlation = numerator / (x_var * y_var) ** 0.5
            else:
                correlation = 0.0
            
            # Determine trend direction
            if correlation > 0.1:
                trend_direction = "improving"
            elif correlation < -0.1:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
            
            trend_strength = abs(correlation)
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        # Calculate statistics
        trend_statistics = {
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores),
            "max_score": max(scores)
        }
        
        return QualityTrend(
            dimension=dimension,
            time_series=time_series,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            statistics=trend_statistics
        )
