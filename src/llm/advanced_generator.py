"""
Advanced Response Generator for NeuroDoc
Implements sophisticated response generation with multi-step reasoning,
contextual awareness, and quality assessment.
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from ..utils.performance import global_performance_monitor, AsyncCache, timed_operation
from ..config import Config

logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of responses the system can generate."""
    DIRECT_ANSWER = "direct_answer"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARY = "summary"
    EXPLANATORY = "explanatory"
    PROCEDURAL = "procedural"


class ResponseComplexity(Enum):
    """Complexity levels for response generation."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
@dataclass
class ResponseContext:
    """Context information for response generation."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    conversation_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    domain_context: Optional[str] = None
    response_type: ResponseType = ResponseType.DIRECT_ANSWER
    complexity_level: ResponseComplexity = ResponseComplexity.MODERATE
    citation_style: str = "academic"
    max_length: int = 1000
    include_sources: bool = True


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    content: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    response_type: ResponseType
    complexity_level: ResponseComplexity
    reasoning_steps: List[str]
    sources_used: List[str]
    generation_time: float
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedResponseGenerator:
    """
    Advanced response generator with multi-step reasoning and quality assessment.
    """
    
    def __init__(self, config: Config):
        """Initialize the advanced response generator."""
        self.config = config
        self.response_templates = self._load_response_templates()
        self.reasoning_strategies = self._initialize_reasoning_strategies()
        
    def _load_response_templates(self) -> Dict[ResponseType, str]:
        """Load response templates for different response types."""
        return {
            ResponseType.DIRECT_ANSWER: """
Based on the provided information, {answer}

{reasoning}

{citations}
""",
            ResponseType.ANALYTICAL: """
Analysis of {topic}:

{analysis_points}

{conclusion}

{citations}
""",
            ResponseType.COMPARATIVE: """
Comparison Analysis:

{comparison_points}

Key Differences:
{differences}

Similarities:
{similarities}

{citations}
""",
            ResponseType.SUMMARY: """
Summary of {topic}:

{key_points}

{conclusion}

{citations}
""",
            ResponseType.EXPLANATORY: """
Explanation of {concept}:

{explanation_steps}

{examples}

{citations}
""",
            ResponseType.PROCEDURAL: """
Procedure for {task}:

{steps}

Important Notes:
{notes}

{citations}
"""
        }
    
    def _initialize_reasoning_strategies(self) -> Dict[str, Any]:
        """Initialize reasoning strategies."""
        return {
            "chain_of_thought": self._chain_of_thought_reasoning,
            "analytical": self._analytical_reasoning,
            "comparative": self._comparative_reasoning,
            "synthesis": self._synthesis_reasoning
        }
    
    @timed_operation("advanced_response_generation", global_performance_monitor)
    async def generate_response(
        self,
        context: ResponseContext,
        llm_client: Any = None
    ) -> GeneratedResponse:
        """
        Generate an advanced response using multi-step reasoning.
        
        Args:
            context: Response generation context
            llm_client: LLM client for generation
            
        Returns:
            Generated response with metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze query and context
            query_analysis = await self._analyze_query(context)
            
            # Step 2: Plan response strategy
            response_plan = await self._plan_response(context, query_analysis)
            
            # Step 3: Execute reasoning
            reasoning_result = await self._execute_reasoning(
                context, response_plan, llm_client
            )
            
            # Step 4: Generate structured response
            response_content = await self._generate_structured_response(
                context, reasoning_result, llm_client
            )
            
            # Step 5: Generate citations
            citations = await self._generate_citations(
                context, reasoning_result
            )
            
            # Step 6: Assess quality
            quality_metrics = await self._assess_response_quality(
                context, response_content, citations
            )
            
            generation_time = time.time() - start_time
            
            response = GeneratedResponse(
                content=response_content,
                citations=citations,
                confidence_score=quality_metrics.get("confidence", 0.0),
                response_type=context.response_type,
                complexity_level=context.complexity_level,
                reasoning_steps=reasoning_result.get("steps", []),
                sources_used=reasoning_result.get("sources", []),
                generation_time=generation_time,
                quality_metrics=quality_metrics,
                metadata={
                    "query_analysis": query_analysis,
                    "response_plan": response_plan,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Generated advanced response in {generation_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating advanced response: {e}")
            raise
    
    async def _analyze_query(self, context: ResponseContext) -> Dict[str, Any]:
        """Analyze the query to understand intent and requirements."""
        analysis = {
            "intent": self._detect_intent(context.query),
            "entities": self._extract_entities(context.query),
            "complexity": self._assess_query_complexity(context.query),
            "information_needs": self._identify_information_needs(context.query),
            "context_requirements": self._analyze_context_requirements(context)
        }
        
        return analysis
    
    def _detect_intent(self, query: str) -> str:
        """Detect the intent of the query."""
        # Intent detection keywords
        intent_keywords = {
            "compare": ["compare", "difference", "versus", "vs", "contrast"],
            "explain": ["explain", "what is", "how does", "why"],
            "summarize": ["summarize", "summary", "overview", "brief"],
            "analyze": ["analyze", "analysis", "examine", "evaluate"],
            "procedure": ["how to", "steps", "procedure", "process", "method"]
        }
        
        query_lower = query.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return "direct_answer"
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from the query."""
        # Simple entity extraction (can be enhanced with NER)
        import re
        
        # Extract capitalized words as potential entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        return list(set(entities))
    
    def _assess_query_complexity(self, query: str) -> ResponseComplexity:
        """Assess the complexity of the query."""
        complexity_indicators = {
            ResponseComplexity.SIMPLE: ["what", "when", "where", "who"],
            ResponseComplexity.MODERATE: ["how", "why", "explain"],
            ResponseComplexity.COMPLEX: ["analyze", "compare", "evaluate", "assess"],
            ResponseComplexity.EXPERT: ["synthesize", "integrate", "critique", "framework"]
        }
        
        query_lower = query.lower()
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return complexity
        
        # Default based on length and structure
        if len(query.split()) > 15:
            return ResponseComplexity.COMPLEX
        elif len(query.split()) > 8:
            return ResponseComplexity.MODERATE
        else:
            return ResponseComplexity.SIMPLE
    
    def _identify_information_needs(self, query: str) -> List[str]:
        """Identify what information is needed to answer the query."""
        needs = []
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["definition", "what is", "meaning"]):
            needs.append("definition")
        
        if any(word in query_lower for word in ["example", "instance", "case"]):
            needs.append("examples")
        
        if any(word in query_lower for word in ["process", "how", "steps"]):
            needs.append("process")
        
        if any(word in query_lower for word in ["compare", "difference", "similar"]):
            needs.append("comparison")
        
        if any(word in query_lower for word in ["benefit", "advantage", "pros"]):
            needs.append("benefits")
        
        if any(word in query_lower for word in ["limitation", "disadvantage", "cons"]):
            needs.append("limitations")
        
        return needs
    
    def _analyze_context_requirements(self, context: ResponseContext) -> Dict[str, Any]:
        """Analyze context requirements for response generation."""
        requirements = {
            "needs_citations": context.include_sources,
            "max_length": context.max_length,
            "citation_style": context.citation_style,
            "domain_specific": context.domain_context is not None,
            "conversational": len(context.conversation_history) > 0,
            "user_preferences": context.user_preferences
        }
        
        return requirements
    
    async def _plan_response(
        self,
        context: ResponseContext,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan the response generation strategy."""
        plan = {
            "reasoning_strategy": self._select_reasoning_strategy(query_analysis),
            "response_structure": self._plan_response_structure(context, query_analysis),
            "information_gathering": self._plan_information_gathering(context, query_analysis),
            "citation_strategy": self._plan_citation_strategy(context)
        }
        
        return plan
    
    def _select_reasoning_strategy(self, query_analysis: Dict[str, Any]) -> str:
        """Select the appropriate reasoning strategy."""
        intent = query_analysis.get("intent", "direct_answer")
        complexity = query_analysis.get("complexity", ResponseComplexity.MODERATE)
        
        if intent == "compare":
            return "comparative"
        elif intent == "analyze":
            return "analytical"
        elif complexity in [ResponseComplexity.COMPLEX, ResponseComplexity.EXPERT]:
            return "synthesis"
        else:
            return "chain_of_thought"
    
    def _plan_response_structure(
        self,
        context: ResponseContext,
        query_analysis: Dict[str, Any]
    ) -> List[str]:
        """Plan the structure of the response."""
        structure = ["introduction"]
        
        information_needs = query_analysis.get("information_needs", [])
        
        if "definition" in information_needs:
            structure.append("definition")
        
        if "process" in information_needs:
            structure.append("process_explanation")
        
        if "comparison" in information_needs:
            structure.append("comparison")
        
        if "examples" in information_needs:
            structure.append("examples")
        
        structure.extend(["main_content", "conclusion"])
        
        if context.include_sources:
            structure.append("citations")
        
        return structure
    
    def _plan_information_gathering(
        self,
        context: ResponseContext,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan how to gather and use information from retrieved chunks."""
        return {
            "prioritize_recent": True,
            "diversity_threshold": 0.7,
            "relevance_threshold": 0.8,
            "max_sources": 5,
            "cross_reference": True
        }
    
    def _plan_citation_strategy(self, context: ResponseContext) -> Dict[str, Any]:
        """Plan the citation strategy."""
        return {
            "style": context.citation_style,
            "inline": True,
            "bibliography": True,
            "max_citations": 10,
            "prefer_primary_sources": True
        }
    
    async def _execute_reasoning(
        self,
        context: ResponseContext,
        response_plan: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Execute the reasoning process."""
        strategy = response_plan["reasoning_strategy"]
        reasoning_function = self.reasoning_strategies.get(strategy)
        
        if reasoning_function:
            return await reasoning_function(context, response_plan, llm_client)
        else:
            return await self._chain_of_thought_reasoning(context, response_plan, llm_client)
    
    async def _chain_of_thought_reasoning(
        self,
        context: ResponseContext,
        response_plan: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning."""
        reasoning_steps = []
        sources_used = []
        
        # Step 1: Information extraction
        relevant_info = self._extract_relevant_information(context.retrieved_chunks)
        reasoning_steps.append("Extracted relevant information from sources")
        sources_used.extend([chunk.get("source", "") for chunk in context.retrieved_chunks])
        
        # Step 2: Information synthesis
        synthesized_info = self._synthesize_information(relevant_info)
        reasoning_steps.append("Synthesized information from multiple sources")
        
        # Step 3: Answer formulation
        reasoning_steps.append("Formulated comprehensive answer")
        
        return {
            "steps": reasoning_steps,
            "sources": list(set(sources_used)),
            "relevant_info": relevant_info,
            "synthesized_info": synthesized_info
        }
    
    async def _analytical_reasoning(
        self,
        context: ResponseContext,
        response_plan: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Execute analytical reasoning."""
        reasoning_steps = []
        sources_used = []
        
        # Step 1: Problem decomposition
        reasoning_steps.append("Decomposed query into analytical components")
        
        # Step 2: Evidence gathering
        evidence = self._gather_evidence(context.retrieved_chunks)
        reasoning_steps.append("Gathered supporting evidence")
        sources_used.extend([chunk.get("source", "") for chunk in context.retrieved_chunks])
        
        # Step 3: Analysis
        analysis = self._perform_analysis(evidence)
        reasoning_steps.append("Performed detailed analysis")
        
        # Step 4: Conclusion drawing
        reasoning_steps.append("Drew evidence-based conclusions")
        
        return {
            "steps": reasoning_steps,
            "sources": list(set(sources_used)),
            "evidence": evidence,
            "analysis": analysis
        }
    
    async def _comparative_reasoning(
        self,
        context: ResponseContext,
        response_plan: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Execute comparative reasoning."""
        reasoning_steps = []
        sources_used = []
        
        # Step 1: Identify comparison subjects
        subjects = self._identify_comparison_subjects(context.query, context.retrieved_chunks)
        reasoning_steps.append("Identified subjects for comparison")
        
        # Step 2: Extract comparable attributes
        attributes = self._extract_comparable_attributes(subjects, context.retrieved_chunks)
        reasoning_steps.append("Extracted comparable attributes")
        sources_used.extend([chunk.get("source", "") for chunk in context.retrieved_chunks])
        
        # Step 3: Perform comparison
        comparison = self._perform_comparison(subjects, attributes)
        reasoning_steps.append("Performed systematic comparison")
        
        return {
            "steps": reasoning_steps,
            "sources": list(set(sources_used)),
            "subjects": subjects,
            "attributes": attributes,
            "comparison": comparison
        }
    
    async def _synthesis_reasoning(
        self,
        context: ResponseContext,
        response_plan: Dict[str, Any],
        llm_client: Any
    ) -> Dict[str, Any]:
        """Execute synthesis reasoning for complex queries."""
        reasoning_steps = []
        sources_used = []
        
        # Step 1: Multi-perspective analysis
        perspectives = self._analyze_multiple_perspectives(context.retrieved_chunks)
        reasoning_steps.append("Analyzed multiple perspectives")
        
        # Step 2: Identify patterns and themes
        patterns = self._identify_patterns(perspectives)
        reasoning_steps.append("Identified patterns and themes")
        sources_used.extend([chunk.get("source", "") for chunk in context.retrieved_chunks])
        
        # Step 3: Synthesize insights
        synthesis = self._synthesize_insights(patterns, perspectives)
        reasoning_steps.append("Synthesized insights from multiple sources")
        
        return {
            "steps": reasoning_steps,
            "sources": list(set(sources_used)),
            "perspectives": perspectives,
            "patterns": patterns,
            "synthesis": synthesis
        }
    
    def _extract_relevant_information(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant information from retrieved chunks."""
        relevant_info = {
            "key_facts": [],
            "concepts": [],
            "relationships": [],
            "examples": []
        }
        
        for chunk in chunks:
            content = chunk.get("content", "")
            
            # Extract key facts (sentences with high importance indicators)
            facts = self._extract_key_facts(content)
            relevant_info["key_facts"].extend(facts)
            
            # Extract concepts
            concepts = self._extract_concepts(content)
            relevant_info["concepts"].extend(concepts)
            
            # Extract relationships
            relationships = self._extract_relationships(content)
            relevant_info["relationships"].extend(relationships)
            
            # Extract examples
            examples = self._extract_examples(content)
            relevant_info["examples"].extend(examples)
        
        return relevant_info
    
    def _extract_key_facts(self, content: str) -> List[str]:
        """Extract key facts from content."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Look for sentences with fact indicators
        fact_indicators = [
            "research shows", "studies indicate", "data reveals",
            "according to", "evidence suggests", "findings show"
        ]
        
        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in fact_indicators):
                facts.append(sentence)
        
        return facts
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        import re
        
        # Extract capitalized terms (potential concepts)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Filter out common words
        common_words = {
            "The", "This", "That", "These", "Those", "However", "Therefore",
            "Furthermore", "Moreover", "Additionally", "Finally"
        }
        
        concepts = [c for c in concepts if c not in common_words]
        
        return list(set(concepts))
    
    def _extract_relationships(self, content: str) -> List[str]:
        """Extract relationships from content."""
        relationship_patterns = [
            r'(\w+)\s+(?:causes?|leads? to|results? in)\s+(\w+)',
            r'(\w+)\s+(?:is related to|correlates with|influences)\s+(\w+)',
            r'(\w+)\s+(?:depends on|requires|needs)\s+(\w+)'
        ]
        
        relationships = []
        for pattern in relationship_patterns:
            import re
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                relationships.append(f"{match[0]} -> {match[1]}")
        
        return relationships
    
    def _extract_examples(self, content: str) -> List[str]:
        """Extract examples from content."""
        import re
        
        example_patterns = [
            r'for example[,:]?\s*([^.!?]*)',
            r'such as\s*([^.!?]*)',
            r'including\s*([^.!?]*)',
            r'e\.g\.?\s*([^.!?]*)'
        ]
        
        examples = []
        for pattern in example_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            examples.extend(matches)
        
        return [example.strip() for example in examples if example.strip()]
    
    def _synthesize_information(self, relevant_info: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize extracted information."""
        synthesis = {
            "main_themes": self._identify_main_themes(relevant_info),
            "supporting_evidence": relevant_info["key_facts"],
            "conceptual_framework": relevant_info["concepts"],
            "illustrative_examples": relevant_info["examples"]
        }
        
        return synthesis
    
    def _identify_main_themes(self, relevant_info: Dict[str, Any]) -> List[str]:
        """Identify main themes from relevant information."""
        # Simple theme identification based on concept frequency
        concept_counts = {}
        for concept in relevant_info["concepts"]:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Return top concepts as themes
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, count in sorted_concepts[:5]]
    
    def _gather_evidence(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gather evidence for analytical reasoning."""
        evidence = {
            "supporting": [],
            "contradicting": [],
            "neutral": []
        }
        
        for chunk in chunks:
            content = chunk.get("content", "")
            evidence_type = self._classify_evidence(content)
            evidence[evidence_type].append({
                "content": content,
                "source": chunk.get("source", ""),
                "confidence": chunk.get("score", 0.0)
            })
        
        return evidence
    
    def _classify_evidence(self, content: str) -> str:
        """Classify evidence as supporting, contradicting, or neutral."""
        # Simple classification based on sentiment indicators
        positive_indicators = ["supports", "confirms", "validates", "proves", "demonstrates"]
        negative_indicators = ["contradicts", "refutes", "disproves", "challenges", "opposes"]
        
        content_lower = content.lower()
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in content_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in content_lower)
        
        if positive_score > negative_score:
            return "supporting"
        elif negative_score > positive_score:
            return "contradicting"
        else:
            return "neutral"
    
    def _perform_analysis(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on gathered evidence."""
        analysis = {
            "strength_of_evidence": self._assess_evidence_strength(evidence),
            "consensus_level": self._assess_consensus(evidence),
            "key_insights": self._extract_insights(evidence),
            "limitations": self._identify_limitations(evidence)
        }
        
        return analysis
    
    def _assess_evidence_strength(self, evidence: Dict[str, Any]) -> str:
        """Assess the strength of evidence."""
        total_evidence = len(evidence["supporting"]) + len(evidence["contradicting"]) + len(evidence["neutral"])
        supporting_ratio = len(evidence["supporting"]) / total_evidence if total_evidence > 0 else 0
        
        if supporting_ratio > 0.8:
            return "strong"
        elif supporting_ratio > 0.6:
            return "moderate"
        elif supporting_ratio > 0.4:
            return "weak"
        else:
            return "insufficient"
    
    def _assess_consensus(self, evidence: Dict[str, Any]) -> str:
        """Assess the level of consensus in evidence."""
        total = len(evidence["supporting"]) + len(evidence["contradicting"]) + len(evidence["neutral"])
        if total == 0:
            return "no_evidence"
        
        contradicting_ratio = len(evidence["contradicting"]) / total
        
        if contradicting_ratio < 0.1:
            return "high_consensus"
        elif contradicting_ratio < 0.3:
            return "moderate_consensus"
        else:
            return "low_consensus"
    
    def _extract_insights(self, evidence: Dict[str, Any]) -> List[str]:
        """Extract key insights from evidence."""
        insights = []
        
        # Analyze supporting evidence for patterns
        supporting_contents = [item["content"] for item in evidence["supporting"]]
        if supporting_contents:
            insights.append("Strong evidence supports the main conclusion")
        
        # Check for contradictions
        if evidence["contradicting"]:
            insights.append("Some contradictory evidence exists")
        
        # Check evidence quality
        high_confidence_evidence = [
            item for item in evidence["supporting"] + evidence["contradicting"] + evidence["neutral"]
            if item.get("confidence", 0) > 0.8
        ]
        
        if len(high_confidence_evidence) > len(evidence["supporting"]) / 2:
            insights.append("Evidence is generally high-quality")
        
        return insights
    
    def _identify_limitations(self, evidence: Dict[str, Any]) -> List[str]:
        """Identify limitations in the evidence."""
        limitations = []
        
        total_evidence = len(evidence["supporting"]) + len(evidence["contradicting"]) + len(evidence["neutral"])
        
        if total_evidence < 3:
            limitations.append("Limited amount of evidence available")
        
        if len(evidence["contradicting"]) > len(evidence["supporting"]) * 0.5:
            limitations.append("Significant contradictory evidence exists")
        
        # Check for source diversity
        sources = set()
        for evidence_list in evidence.values():
            for item in evidence_list:
                sources.add(item.get("source", ""))
        
        if len(sources) < 3:
            limitations.append("Limited source diversity")
        
        return limitations
    
    def _identify_comparison_subjects(self, query: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """Identify subjects for comparison."""
        # Extract potential subjects from query
        import re
        
        # Look for "X vs Y", "X and Y", "compare X with Y" patterns
        vs_pattern = r'(\w+(?:\s+\w+)*)\s+(?:vs|versus|compared to|against)\s+(\w+(?:\s+\w+)*)'
        and_pattern = r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)'
        
        subjects = []
        
        vs_match = re.search(vs_pattern, query, re.IGNORECASE)
        if vs_match:
            subjects.extend([vs_match.group(1).strip(), vs_match.group(2).strip()])
        
        and_match = re.search(and_pattern, query, re.IGNORECASE)
        if and_match and not vs_match:  # Avoid duplicates
            subjects.extend([and_match.group(1).strip(), and_match.group(2).strip()])
        
        # If no clear subjects found, extract from chunks
        if not subjects:
            subjects = self._extract_subjects_from_chunks(chunks)
        
        return subjects[:2]  # Limit to 2 subjects for binary comparison
    
    def _extract_subjects_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract comparison subjects from chunks."""
        subjects = []
        
        for chunk in chunks:
            content = chunk.get("content", "")
            # Extract capitalized terms as potential subjects
            import re
            terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            subjects.extend(terms)
        
        # Return most frequent terms
        from collections import Counter
        term_counts = Counter(subjects)
        return [term for term, count in term_counts.most_common(5)]
    
    def _extract_comparable_attributes(self, subjects: List[str], chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract attributes that can be compared between subjects."""
        attributes = []
        
        # Common comparison attributes
        common_attributes = [
            "performance", "efficiency", "cost", "benefits", "limitations",
            "features", "characteristics", "advantages", "disadvantages"
        ]
        
        # Extract attributes mentioned in chunks
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            for attr in common_attributes:
                if attr in content:
                    attributes.append(attr)
        
        return list(set(attributes))
    
    def _perform_comparison(self, subjects: List[str], attributes: List[str]) -> Dict[str, Any]:
        """Perform comparison between subjects on given attributes."""
        comparison = {
            "subjects": subjects,
            "attributes": attributes,
            "similarities": [],
            "differences": [],
            "summary": ""
        }
        
        # This is a simplified comparison - in practice, would use more sophisticated analysis
        if len(subjects) >= 2:
            comparison["summary"] = f"Comparison between {subjects[0]} and {subjects[1]}"
            
            # Add placeholder similarities and differences
            comparison["similarities"].append("Both are discussed in the context of the query")
            comparison["differences"].append("Specific differences based on available evidence")
        
        return comparison
    
    def _analyze_multiple_perspectives(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple perspectives from different sources."""
        perspectives = {
            "viewpoints": [],
            "consensus_areas": [],
            "disagreement_areas": [],
            "evidence_quality": {}
        }
        
        for i, chunk in enumerate(chunks):
            perspective = {
                "source": chunk.get("source", f"Source {i+1}"),
                "main_points": self._extract_main_points(chunk.get("content", "")),
                "stance": self._determine_stance(chunk.get("content", "")),
                "confidence": chunk.get("score", 0.0)
            }
            perspectives["viewpoints"].append(perspective)
        
        return perspectives
    
    def _extract_main_points(self, content: str) -> List[str]:
        """Extract main points from content."""
        import re
        
        # Split into sentences and identify important ones
        sentences = re.split(r'[.!?]+', content)
        main_points = []
        
        importance_indicators = [
            "important", "key", "main", "primary", "significant",
            "crucial", "essential", "fundamental", "critical"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in importance_indicators):
                main_points.append(sentence)
        
        # If no important sentences found, take first few sentences
        if not main_points and sentences:
            main_points = [s.strip() for s in sentences[:3] if s.strip()]
        
        return main_points
    
    def _determine_stance(self, content: str) -> str:
        """Determine the stance/position of the content."""
        positive_indicators = ["benefit", "advantage", "positive", "effective", "successful"]
        negative_indicators = ["limitation", "disadvantage", "negative", "ineffective", "problematic"]
        
        content_lower = content.lower()
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in content_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in content_lower)
        
        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"
    
    def _identify_patterns(self, perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns across perspectives."""
        patterns = {
            "recurring_themes": [],
            "stance_distribution": {},
            "confidence_patterns": {},
            "source_agreements": []
        }
        
        # Analyze stance distribution
        stances = [vp["stance"] for vp in perspectives["viewpoints"]]
        from collections import Counter
        patterns["stance_distribution"] = dict(Counter(stances))
        
        # Analyze confidence patterns
        confidences = [vp["confidence"] for vp in perspectives["viewpoints"]]
        patterns["confidence_patterns"] = {
            "average": sum(confidences) / len(confidences) if confidences else 0,
            "high_confidence_count": len([c for c in confidences if c > 0.8]),
            "low_confidence_count": len([c for c in confidences if c < 0.5])
        }
        
        return patterns
    
    def _synthesize_insights(self, patterns: Dict[str, Any], perspectives: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights from patterns and perspectives."""
        synthesis = {
            "key_insights": [],
            "consensus_level": "unknown",
            "reliability_assessment": "unknown",
            "actionable_conclusions": []
        }
        
        # Assess consensus
        stance_dist = patterns.get("stance_distribution", {})
        total_viewpoints = sum(stance_dist.values())
        
        if total_viewpoints > 0:
            max_stance_count = max(stance_dist.values())
            consensus_ratio = max_stance_count / total_viewpoints
            
            if consensus_ratio > 0.8:
                synthesis["consensus_level"] = "high"
            elif consensus_ratio > 0.6:
                synthesis["consensus_level"] = "moderate"
            else:
                synthesis["consensus_level"] = "low"
        
        # Assess reliability
        conf_patterns = patterns.get("confidence_patterns", {})
        avg_confidence = conf_patterns.get("average", 0)
        
        if avg_confidence > 0.8:
            synthesis["reliability_assessment"] = "high"
        elif avg_confidence > 0.6:
            synthesis["reliability_assessment"] = "moderate"
        else:
            synthesis["reliability_assessment"] = "low"
        
        # Generate insights
        synthesis["key_insights"].append(f"Consensus level: {synthesis['consensus_level']}")
        synthesis["key_insights"].append(f"Evidence reliability: {synthesis['reliability_assessment']}")
        
        return synthesis
    
    async def _generate_structured_response(
        self,
        context: ResponseContext,
        reasoning_result: Dict[str, Any],
        llm_client: Any
    ) -> str:
        """Generate structured response content."""
        # Get the appropriate template
        template = self.response_templates.get(context.response_type, self.response_templates[ResponseType.DIRECT_ANSWER])
        
        # Prepare template variables
        template_vars = self._prepare_template_variables(context, reasoning_result)
        
        # Generate response using template
        response_content = template.format(**template_vars)
        
        # Ensure response length constraints
        if len(response_content) > context.max_length:
            response_content = self._truncate_response(response_content, context.max_length)
        
        return response_content
    
    def _prepare_template_variables(self, context: ResponseContext, reasoning_result: Dict[str, Any]) -> Dict[str, str]:
        """Prepare variables for response template."""
        variables = {
            "answer": "Based on the available information",
            "reasoning": "\n".join(f"• {step}" for step in reasoning_result.get("steps", [])),
            "citations": "[Citations will be added]",  # Placeholder
            "topic": "the requested topic",
            "analysis_points": "Key analysis points from the reasoning process",
            "conclusion": "Conclusion based on the analysis",
            "comparison_points": "Comparison analysis",
            "differences": "Key differences identified",
            "similarities": "Similarities found",
            "key_points": "Main points from the analysis",
            "concept": "the concept in question",
            "explanation_steps": "Step-by-step explanation",
            "examples": "Relevant examples",
            "task": "the requested task",
            "steps": "Procedural steps",
            "notes": "Important notes and considerations"
        }
        
        # Customize based on reasoning result
        if "synthesized_info" in reasoning_result:
            synth_info = reasoning_result["synthesized_info"]
            variables["answer"] = self._format_main_themes(synth_info.get("main_themes", []))
            variables["key_points"] = self._format_key_points(synth_info.get("supporting_evidence", []))
        
        if "analysis" in reasoning_result:
            analysis = reasoning_result["analysis"]
            variables["analysis_points"] = self._format_analysis_points(analysis)
            variables["conclusion"] = analysis.get("key_insights", ["Analysis complete"])[0]
        
        if "comparison" in reasoning_result:
            comparison = reasoning_result["comparison"]
            variables["comparison_points"] = comparison.get("summary", "Comparison analysis")
            variables["differences"] = "\n".join(f"• {diff}" for diff in comparison.get("differences", []))
            variables["similarities"] = "\n".join(f"• {sim}" for sim in comparison.get("similarities", []))
        
        return variables
    
    def _format_main_themes(self, themes: List[str]) -> str:
        """Format main themes for response."""
        if not themes:
            return "The analysis reveals several key points."
        
        return f"The main themes that emerge are: {', '.join(themes)}."
    
    def _format_key_points(self, evidence: List[str]) -> str:
        """Format key points for response."""
        if not evidence:
            return "Key points from the analysis."
        
        formatted_points = []
        for i, point in enumerate(evidence[:5], 1):  # Limit to top 5 points
            formatted_points.append(f"{i}. {point}")
        
        return "\n".join(formatted_points)
    
    def _format_analysis_points(self, analysis: Dict[str, Any]) -> str:
        """Format analysis points for response."""
        points = []
        
        if analysis.get("strength_of_evidence"):
            points.append(f"Evidence strength: {analysis['strength_of_evidence']}")
        
        if analysis.get("consensus_level"):
            points.append(f"Consensus level: {analysis['consensus_level']}")
        
        if analysis.get("key_insights"):
            points.extend(analysis["key_insights"])
        
        return "\n".join(f"• {point}" for point in points)
    
    def _truncate_response(self, response: str, max_length: int) -> str:
        """Truncate response to fit length constraints."""
        if len(response) <= max_length:
            return response
        
        # Find a good breaking point near the limit
        truncated = response[:max_length-100]  # Leave room for ellipsis and conclusion
        
        # Find last complete sentence
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we find a period reasonably close to the end
            truncated = truncated[:last_period + 1]
        
        truncated += "\n\n[Response truncated due to length constraints]"
        
        return truncated
    
    async def _generate_citations(
        self,
        context: ResponseContext,
        reasoning_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate citations for the response."""
        if not context.include_sources:
            return []
        
        citations = []
        sources_used = reasoning_result.get("sources", [])
        
        # Generate citations from retrieved chunks
        for i, chunk in enumerate(context.retrieved_chunks):
            if i >= 10:  # Limit to 10 citations
                break
                
            citation = {
                "id": i + 1,
                "source": chunk.get("source", f"Source {i+1}"),
                "content_preview": chunk.get("content", "")[:200] + "...",
                "relevance_score": chunk.get("score", 0.0),
                "page": chunk.get("page", None),
                "section": chunk.get("section", None),
                "url": chunk.get("url", None)
            }
            
            citations.append(citation)
        
        return citations
    
    async def _assess_response_quality(
        self,
        context: ResponseContext,
        response_content: str,
        citations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess the quality of the generated response."""
        quality_metrics = {}
        
        # Completeness (how well the response addresses the query)
        quality_metrics["completeness"] = self._assess_completeness(context.query, response_content)
        
        # Accuracy (based on source quality and citation strength)
        quality_metrics["accuracy"] = self._assess_accuracy(citations)
        
        # Clarity (readability and structure)
        quality_metrics["clarity"] = self._assess_clarity(response_content)
        
        # Relevance (how relevant the response is to the query)
        quality_metrics["relevance"] = self._assess_relevance(context.query, response_content)
        
        # Citation quality (quality and appropriateness of citations)
        quality_metrics["citation_quality"] = self._assess_citation_quality(citations)
        
        # Overall confidence (weighted average)
        weights = {
            "completeness": 0.25,
            "accuracy": 0.25,
            "clarity": 0.15,
            "relevance": 0.25,
            "citation_quality": 0.10
        }
        
        quality_metrics["confidence"] = sum(
            quality_metrics[metric] * weight
            for metric, weight in weights.items()
            if metric in quality_metrics
        )
        
        return quality_metrics
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess how completely the response addresses the query."""
        # Simple completeness assessment based on length and query keywords
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Check keyword coverage
        keyword_coverage = len(query_words.intersection(response_words)) / len(query_words) if query_words else 0
        
        # Check response length adequacy
        length_score = min(len(response) / 500, 1.0)  # Assume 500 chars is adequate length
        
        # Combined score
        completeness = (keyword_coverage * 0.6) + (length_score * 0.4)
        
        return min(completeness, 1.0)
    
    def _assess_accuracy(self, citations: List[Dict[str, Any]]) -> float:
        """Assess accuracy based on citation quality."""
        if not citations:
            return 0.5  # Neutral score if no citations
        
        # Average relevance score of citations
        relevance_scores = [citation.get("relevance_score", 0.0) for citation in citations]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Number of citations (more citations can indicate better support)
        citation_count_score = min(len(citations) / 5, 1.0)  # Ideal: 5+ citations
        
        # Combined accuracy score
        accuracy = (avg_relevance * 0.7) + (citation_count_score * 0.3)
        
        return accuracy
    
    def _assess_clarity(self, response: str) -> float:
        """Assess the clarity and readability of the response."""
        # Simple clarity metrics
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Penalize very long or very short sentences
        if avg_sentence_length < 5 or avg_sentence_length > 30:
            length_score = 0.5
        else:
            length_score = 1.0
        
        # Check for structure indicators (bullet points, numbered lists, etc.)
        structure_indicators = ['•', '1.', '2.', '3.', '-', '*']
        has_structure = any(indicator in response for indicator in structure_indicators)
        structure_score = 1.0 if has_structure else 0.7
        
        # Combined clarity score
        clarity = (length_score * 0.4) + (structure_score * 0.6)
        
        return clarity
    
    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess how relevant the response is to the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Jaccard similarity
        intersection = query_words.intersection(response_words)
        union = query_words.union(response_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0
        
        # Boost score if query keywords appear in response
        query_keywords_in_response = len(intersection) / len(query_words) if query_words else 0
        
        # Combined relevance score
        relevance = (jaccard_similarity * 0.4) + (query_keywords_in_response * 0.6)
        
        return min(relevance, 1.0)
    
    def _assess_citation_quality(self, citations: List[Dict[str, Any]]) -> float:
        """Assess the quality of citations."""
        if not citations:
            return 0.0
        
        # Check citation completeness (has source, relevance score, etc.)
        complete_citations = 0
        for citation in citations:
            if citation.get("source") and citation.get("relevance_score", 0) > 0:
                complete_citations += 1
        
        completeness_score = complete_citations / len(citations)
        
        # Check average relevance of citations
        relevance_scores = [citation.get("relevance_score", 0.0) for citation in citations]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Combined citation quality
        citation_quality = (completeness_score * 0.5) + (avg_relevance * 0.5)
        
        return citation_quality
