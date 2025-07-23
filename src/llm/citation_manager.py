"""
Enhanced Citation Manager for NeuroDoc
Implements sophisticated citation generation, verification, and formatting
with support for multiple citation styles and source validation.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import hashlib
from datetime import datetime
from urllib.parse import urlparse

from ..utils.performance import global_performance_monitor, AsyncCache, timed_operation
from ..config import Config

logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Supported citation styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"
    ACADEMIC = "academic"
    SIMPLE = "simple"


class SourceType(Enum):
    """Types of sources that can be cited."""
    ACADEMIC_PAPER = "academic_paper"
    BOOK = "book"
    CHAPTER = "chapter"
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    WEBSITE = "website"
    REPORT = "report"
    THESIS = "thesis"
    PATENT = "patent"
    DATASET = "dataset"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class CitationQuality(Enum):
    """Quality levels for citations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class SourceMetadata:
    """Metadata for a source."""
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    institution: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    access_date: Optional[str] = None
    language: str = "en"
    quality_indicators: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A structured citation with metadata and formatting."""
    id: str
    source_metadata: SourceMetadata
    content_snippet: str
    page_reference: Optional[str] = None
    section_reference: Optional[str] = None
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 0.0
    citation_text: str = ""
    inline_citation: str = ""
    context_before: str = ""
    context_after: str = ""
    verification_status: str = "unverified"
    extracted_claims: List[str] = field(default_factory=list)
    supporting_quotes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationGroup:
    """A group of related citations."""
    topic: str
    citations: List[Citation]
    group_confidence: float = 0.0
    consensus_level: float = 0.0
    conflicting_information: List[str] = field(default_factory=list)
    synthesis: str = ""


@dataclass
class Bibliography:
    """A complete bibliography with citations organized by style."""
    citations: List[Citation]
    style: CitationStyle
    formatted_entries: List[str] = field(default_factory=list)
    grouped_citations: List[CitationGroup] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class CitationManager:
    """
    Enhanced citation manager with advanced features for citation generation,
    verification, and quality assessment.
    """
    
    def __init__(self, config: Config):
        """Initialize the citation manager."""
        self.config = config
        self.citation_styles = self._initialize_citation_styles()
        self.source_extractors = self._initialize_source_extractors()
        self.quality_assessors = self._initialize_quality_assessors()
        self.citation_cache = {}
        self.verification_cache = {}
        
    def _initialize_citation_styles(self) -> Dict[CitationStyle, Dict[str, str]]:
        """Initialize citation style templates."""
        return {
            CitationStyle.APA: {
                "book": "{authors} ({year}). {title}. {publisher}.",
                "journal": "{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}.",
                "website": "{authors} ({year}). {title}. Retrieved from {url}",
                "intext": "({authors}, {year})",
                "intext_page": "({authors}, {year}, p. {page})"
            },
            CitationStyle.MLA: {
                "book": "{authors}. {title}. {publisher}, {year}.",
                "journal": "{authors}. \"{title}.\" {journal}, vol. {volume}, no. {issue}, {year}, pp. {pages}.",
                "website": "{authors}. \"{title}.\" {website}, {date}, {url}.",
                "intext": "({authors} {page})",
                "intext_nopage": "({authors})"
            },
            CitationStyle.CHICAGO: {
                "book": "{authors}. {title}. {place}: {publisher}, {year}.",
                "journal": "{authors}. \"{title}.\" {journal} {volume}, no. {issue} ({year}): {pages}.",
                "website": "{authors}. \"{title}.\" Accessed {access_date}. {url}.",
                "footnote": "{authors}, {title} ({place}: {publisher}, {year}), {page}."
            },
            CitationStyle.IEEE: {
                "journal": "[{number}] {authors}, \"{title},\" {journal}, vol. {volume}, no. {issue}, pp. {pages}, {year}.",
                "book": "[{number}] {authors}, {title}. {place}: {publisher}, {year}.",
                "website": "[{number}] {authors}, \"{title},\" {website}. [Online]. Available: {url}. [Accessed: {access_date}].",
                "intext": "[{number}]"
            },
            CitationStyle.ACADEMIC: {
                "general": "{authors} ({year}). {title}. {source_details}",
                "intext": "({authors}, {year})",
                "intext_page": "({authors}, {year}: {page})"
            },
            CitationStyle.SIMPLE: {
                "general": "{title} - {authors} ({year})",
                "intext": "[{number}]"
            }
        }
    
    def _initialize_source_extractors(self) -> Dict[str, callable]:
        """Initialize source metadata extractors."""
        return {
            "doi": self._extract_doi_metadata,
            "isbn": self._extract_isbn_metadata,
            "url": self._extract_url_metadata,
            "text": self._extract_text_metadata,
            "filename": self._extract_filename_metadata
        }
    
    def _initialize_quality_assessors(self) -> Dict[str, callable]:
        """Initialize quality assessment functions."""
        return {
            "source_authority": self._assess_source_authority,
            "publication_quality": self._assess_publication_quality,
            "citation_accuracy": self._assess_citation_accuracy,
            "relevance": self._assess_citation_relevance,
            "freshness": self._assess_content_freshness
        }
    
    @timed_operation("citation_generation", global_performance_monitor)
    async def generate_citations(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        query: str,
        style: CitationStyle = CitationStyle.ACADEMIC,
        max_citations: int = 10,
        quality_threshold: float = 0.5
    ) -> List[Citation]:
        """
        Generate high-quality citations from retrieved chunks.
        
        Args:
            retrieved_chunks: Retrieved document chunks
            query: Original query for relevance assessment
            style: Citation style to use
            max_citations: Maximum number of citations
            quality_threshold: Minimum quality score threshold
            
        Returns:
            List of generated citations
        """
        try:
            citations = []
            
            for i, chunk in enumerate(retrieved_chunks[:max_citations * 2]):  # Get extra for filtering
                # Extract source metadata
                source_metadata = await self._extract_source_metadata(chunk)
                
                # Create citation
                citation = await self._create_citation(
                    chunk, source_metadata, query, style, i + 1
                )
                
                # Assess citation quality
                citation.quality_score = await self._assess_citation_quality(citation)
                
                # Filter by quality threshold
                if citation.quality_score >= quality_threshold:
                    citations.append(citation)
                
                if len(citations) >= max_citations:
                    break
            
            # Sort by quality and relevance
            citations.sort(key=lambda c: (c.quality_score, c.relevance_score), reverse=True)
            
            # Verify citations
            await self._verify_citations(citations)
            
            logger.info(f"Generated {len(citations)} high-quality citations")
            return citations[:max_citations]
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            raise
    
    @timed_operation("bibliography_creation", global_performance_monitor)
    async def create_bibliography(
        self,
        citations: List[Citation],
        style: CitationStyle = CitationStyle.ACADEMIC,
        group_by_topic: bool = True
    ) -> Bibliography:
        """
        Create a formatted bibliography from citations.
        
        Args:
            citations: List of citations
            style: Citation style for formatting
            group_by_topic: Whether to group citations by topic
            
        Returns:
            Formatted bibliography
        """
        try:
            # Format individual citations
            formatted_entries = []
            for citation in citations:
                formatted_entry = await self._format_citation(citation, style)
                formatted_entries.append(formatted_entry)
            
            # Group citations if requested
            grouped_citations = []
            if group_by_topic:
                grouped_citations = await self._group_citations_by_topic(citations)
            
            # Calculate statistics
            statistics = self._calculate_bibliography_statistics(citations)
            
            bibliography = Bibliography(
                citations=citations,
                style=style,
                formatted_entries=formatted_entries,
                grouped_citations=grouped_citations,
                statistics=statistics
            )
            
            logger.info(f"Created bibliography with {len(citations)} citations")
            return bibliography
            
        except Exception as e:
            logger.error(f"Error creating bibliography: {e}")
            raise
    
    async def _extract_source_metadata(self, chunk: Dict[str, Any]) -> SourceMetadata:
        """Extract comprehensive source metadata from a chunk."""
        metadata = SourceMetadata()
        
        # Basic information from chunk
        metadata.title = chunk.get("title", "")
        metadata.url = chunk.get("url", "")
        
        # Extract from source field
        source = chunk.get("source", "")
        if source:
            metadata = await self._enhance_metadata_from_source(metadata, source)
        
        # Extract from content
        content = chunk.get("content", "")
        if content:
            metadata = await self._enhance_metadata_from_content(metadata, content)
        
        # Determine source type
        metadata.source_type = self._determine_source_type(metadata)
        
        # Set access date
        metadata.access_date = datetime.now().strftime("%Y-%m-%d")
        
        return metadata
    
    async def _enhance_metadata_from_source(
        self,
        metadata: SourceMetadata,
        source: str
    ) -> SourceMetadata:
        """Enhance metadata from source string."""
        # Try different extraction methods
        for extractor_name, extractor_func in self.source_extractors.items():
            try:
                enhanced_metadata = await extractor_func(source, metadata)
                if enhanced_metadata:
                    metadata = enhanced_metadata
                    break
            except Exception as e:
                logger.debug(f"Extractor {extractor_name} failed for source {source}: {e}")
        
        return metadata
    
    async def _enhance_metadata_from_content(
        self,
        metadata: SourceMetadata,
        content: str
    ) -> SourceMetadata:
        """Enhance metadata from content text."""
        # Extract title if not present
        if not metadata.title:
            metadata.title = self._extract_title_from_content(content)
        
        # Extract authors
        authors = self._extract_authors_from_content(content)
        if authors:
            metadata.authors = authors
        
        # Extract publication date
        pub_date = self._extract_date_from_content(content)
        if pub_date:
            metadata.publication_date = pub_date
        
        # Extract DOI
        doi = self._extract_doi_from_content(content)
        if doi:
            metadata.doi = doi
        
        return metadata
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from content text."""
        # Look for title patterns
        title_patterns = [
            r'^([A-Z][^.!?]*[.!?])',  # First sentence starting with capital
            r'Title:\s*([^\n]+)',      # Explicit title field
            r'^([^:]+):',              # Text before first colon
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content.strip(), re.MULTILINE)
            if match:
                title = match.group(1).strip()
                if len(title) > 10 and len(title) < 200:  # Reasonable title length
                    return title
        
        # Fallback: first meaningful sentence
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:
                return sentence
        
        return None
    
    def _extract_authors_from_content(self, content: str) -> List[str]:
        """Extract author names from content."""
        author_patterns = [
            r'Authors?:\s*([^\n]+)',
            r'By\s+([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
            r'([A-Z][a-z]+,?\s+[A-Z]\.(?:\s*[A-Z]\.)?(?:\s*[A-Z][a-z]+)?)',  # Last, F. M.
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, str):
                    # Split multiple authors
                    author_list = re.split(r',\s*(?:and\s+)?|(?:\s+and\s+)', match)
                    for author in author_list:
                        author = author.strip()
                        if author and len(author) > 3:
                            authors.append(author)
        
        return list(set(authors))  # Remove duplicates
    
    def _extract_date_from_content(self, content: str) -> Optional[str]:
        """Extract publication date from content."""
        date_patterns = [
            r'(\d{4})',  # Simple year
            r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_doi_from_content(self, content: str) -> Optional[str]:
        """Extract DOI from content."""
        doi_pattern = r'doi:\s*(10\.\d+/[^\s]+)'
        match = re.search(doi_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _determine_source_type(self, metadata: SourceMetadata) -> SourceType:
        """Determine the type of source based on metadata."""
        # Check for academic paper indicators
        if metadata.doi or metadata.journal:
            return SourceType.JOURNAL_ARTICLE
        
        # Check for book indicators
        if metadata.isbn or "book" in (metadata.title or "").lower():
            return SourceType.BOOK
        
        # Check for website indicators
        if metadata.url and not metadata.journal:
            return SourceType.WEBSITE
        
        # Check for conference paper indicators
        if any(keyword in (metadata.title or "").lower() 
               for keyword in ["conference", "proceedings", "symposium"]):
            return SourceType.CONFERENCE_PAPER
        
        # Check for report indicators
        if any(keyword in (metadata.title or "").lower() 
               for keyword in ["report", "technical report", "white paper"]):
            return SourceType.REPORT
        
        return SourceType.UNKNOWN
    
    async def _create_citation(
        self,
        chunk: Dict[str, Any],
        metadata: SourceMetadata,
        query: str,
        style: CitationStyle,
        citation_number: int
    ) -> Citation:
        """Create a citation object from chunk and metadata."""
        # Generate unique citation ID
        citation_id = hashlib.md5(
            f"{metadata.title}{metadata.url}{chunk.get('content', '')[:100]}".encode()
        ).hexdigest()[:8]
        
        # Extract content snippet
        content_snippet = chunk.get("content", "")[:500]
        
        # Calculate relevance score
        relevance_score = chunk.get("score", 0.0)
        
        # Generate supporting quotes
        supporting_quotes = self._extract_supporting_quotes(
            chunk.get("content", ""), query
        )
        
        # Extract claims
        extracted_claims = self._extract_claims_from_content(
            chunk.get("content", "")
        )
        
        # Generate formatted citation text
        citation_text = await self._format_citation_text(metadata, style)
        
        # Generate inline citation
        inline_citation = await self._format_inline_citation(
            metadata, style, citation_number
        )
        
        citation = Citation(
            id=citation_id,
            source_metadata=metadata,
            content_snippet=content_snippet,
            page_reference=chunk.get("page"),
            section_reference=chunk.get("section"),
            relevance_score=relevance_score,
            citation_text=citation_text,
            inline_citation=inline_citation,
            extracted_claims=extracted_claims,
            supporting_quotes=supporting_quotes,
            metadata={
                "chunk_id": chunk.get("id"),
                "original_score": chunk.get("score", 0.0),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return citation
    
    def _extract_supporting_quotes(self, content: str, query: str) -> List[str]:
        """Extract quotes that support the query."""
        quotes = []
        
        # Simple approach: find sentences containing query keywords
        query_words = set(query.lower().split())
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Minimum sentence length
                sentence_words = set(sentence.lower().split())
                overlap = query_words.intersection(sentence_words)
                
                # If sentence contains query keywords, consider it supporting
                if len(overlap) >= min(2, len(query_words) // 2):
                    quotes.append(sentence)
        
        return quotes[:3]  # Limit to top 3 quotes
    
    def _extract_claims_from_content(self, content: str) -> List[str]:
        """Extract factual claims from content."""
        claims = []
        
        # Look for sentences with claim indicators
        claim_indicators = [
            "research shows", "studies indicate", "data reveals",
            "findings suggest", "evidence demonstrates", "results show",
            "analysis reveals", "experiments prove"
        ]
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                claims.append(sentence)
        
        return claims[:5]  # Limit to top 5 claims
    
    async def _format_citation_text(
        self,
        metadata: SourceMetadata,
        style: CitationStyle
    ) -> str:
        """Format citation text according to style."""
        style_templates = self.citation_styles.get(style, {})
        
        # Prepare variables for template
        variables = {
            "authors": self._format_authors(metadata.authors, style),
            "year": metadata.publication_date or "n.d.",
            "title": metadata.title or "Untitled",
            "journal": metadata.journal or "",
            "volume": metadata.volume or "",
            "issue": metadata.issue or "",
            "pages": metadata.pages or "",
            "publisher": metadata.publisher or "",
            "url": metadata.url or "",
            "doi": metadata.doi or "",
            "access_date": metadata.access_date or ""
        }
        
        # Select appropriate template based on source type
        template_key = self._get_template_key(metadata.source_type)
        template = style_templates.get(template_key, style_templates.get("general", "{title} - {authors} ({year})"))
        
        # Format citation
        try:
            formatted_citation = template.format(**variables)
            return formatted_citation
        except KeyError as e:
            logger.warning(f"Missing variable for citation formatting: {e}")
            # Fallback formatting
            return f"{variables['title']} - {variables['authors']} ({variables['year']})"
    
    def _format_authors(self, authors: List[str], style: CitationStyle) -> str:
        """Format author list according to style."""
        if not authors:
            return "Anonymous"
        
        if style == CitationStyle.APA:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} & {authors[1]}"
            else:
                return f"{authors[0]} et al."
        
        elif style == CitationStyle.MLA:
            if len(authors) == 1:
                return authors[0]
            elif len(authors) == 2:
                return f"{authors[0]} and {authors[1]}"
            else:
                return f"{authors[0]} et al."
        
        else:  # Default formatting
            if len(authors) == 1:
                return authors[0]
            elif len(authors) <= 3:
                return ", ".join(authors[:-1]) + f" and {authors[-1]}"
            else:
                return f"{authors[0]} et al."
    
    def _get_template_key(self, source_type: SourceType) -> str:
        """Get template key for source type."""
        type_mapping = {
            SourceType.JOURNAL_ARTICLE: "journal",
            SourceType.BOOK: "book",
            SourceType.WEBSITE: "website",
            SourceType.CONFERENCE_PAPER: "journal",  # Similar to journal
            SourceType.REPORT: "book",  # Similar to book
        }
        return type_mapping.get(source_type, "general")
    
    async def _format_inline_citation(
        self,
        metadata: SourceMetadata,
        style: CitationStyle,
        number: int
    ) -> str:
        """Format inline citation according to style."""
        style_templates = self.citation_styles.get(style, {})
        
        variables = {
            "authors": self._format_authors_inline(metadata.authors, style),
            "year": metadata.publication_date or "n.d.",
            "number": number,
            "page": ""  # Will be filled when specific page is referenced
        }
        
        # Get inline template
        if style in [CitationStyle.IEEE, CitationStyle.SIMPLE]:
            template = style_templates.get("intext", "[{number}]")
        else:
            template = style_templates.get("intext", "({authors}, {year})")
        
        try:
            return template.format(**variables)
        except KeyError:
            return f"[{number}]"  # Fallback
    
    def _format_authors_inline(self, authors: List[str], style: CitationStyle) -> str:
        """Format authors for inline citation."""
        if not authors:
            return "Anonymous"
        
        # For inline citations, usually just first author or first author et al.
        if len(authors) == 1:
            # Extract last name
            parts = authors[0].split()
            return parts[-1] if parts else authors[0]
        else:
            parts = authors[0].split()
            first_author = parts[-1] if parts else authors[0]
            return f"{first_author} et al."
    
    async def _assess_citation_quality(self, citation: Citation) -> float:
        """Assess the overall quality of a citation."""
        quality_scores = {}
        
        # Run all quality assessments
        for assessor_name, assessor_func in self.quality_assessors.items():
            try:
                score = await assessor_func(citation)
                quality_scores[assessor_name] = score
            except Exception as e:
                logger.debug(f"Quality assessor {assessor_name} failed: {e}")
                quality_scores[assessor_name] = 0.5  # Neutral score
        
        # Calculate weighted average
        weights = {
            "source_authority": 0.25,
            "publication_quality": 0.25,
            "citation_accuracy": 0.20,
            "relevance": 0.20,
            "freshness": 0.10
        }
        
        overall_quality = sum(
            quality_scores.get(metric, 0.5) * weight
            for metric, weight in weights.items()
        )
        
        return min(1.0, max(0.0, overall_quality))
    
    async def _assess_source_authority(self, citation: Citation) -> float:
        """Assess the authority of the source."""
        metadata = citation.source_metadata
        score = 0.5  # Base score
        
        # Academic sources get higher scores
        if metadata.source_type in [SourceType.JOURNAL_ARTICLE, SourceType.ACADEMIC_PAPER]:
            score += 0.3
        
        # Presence of DOI indicates peer review
        if metadata.doi:
            score += 0.2
        
        # Known academic institutions
        if metadata.institution and any(
            keyword in metadata.institution.lower()
            for keyword in ["university", "institute", "research", "academic"]
        ):
            score += 0.1
        
        # Reputable publishers
        if metadata.publisher and any(
            keyword in metadata.publisher.lower()
            for keyword in ["springer", "elsevier", "ieee", "acm", "nature", "science"]
        ):
            score += 0.1
        
        return min(1.0, score)
    
    async def _assess_publication_quality(self, citation: Citation) -> float:
        """Assess the quality of the publication."""
        metadata = citation.source_metadata
        score = 0.5  # Base score
        
        # Peer-reviewed indicators
        if metadata.doi or metadata.journal:
            score += 0.2
        
        # Complete metadata indicates higher quality
        completeness_score = sum([
            1 if metadata.title else 0,
            1 if metadata.authors else 0,
            1 if metadata.publication_date else 0,
            1 if metadata.journal or metadata.publisher else 0
        ]) / 4
        
        score += completeness_score * 0.3
        
        return min(1.0, score)
    
    async def _assess_citation_accuracy(self, citation: Citation) -> float:
        """Assess the accuracy of the citation formatting."""
        score = 0.5  # Base score
        
        # Check if essential elements are present
        if citation.citation_text and len(citation.citation_text) > 10:
            score += 0.2
        
        if citation.inline_citation:
            score += 0.1
        
        # Check metadata completeness
        metadata = citation.source_metadata
        if metadata.title and metadata.authors:
            score += 0.2
        
        return min(1.0, score)
    
    async def _assess_citation_relevance(self, citation: Citation) -> float:
        """Assess how relevant the citation is."""
        # Use the relevance score from retrieval
        base_relevance = citation.relevance_score
        
        # Boost if there are supporting quotes
        if citation.supporting_quotes:
            base_relevance += 0.1
        
        # Boost if there are extracted claims
        if citation.extracted_claims:
            base_relevance += 0.1
        
        return min(1.0, base_relevance)
    
    async def _assess_content_freshness(self, citation: Citation) -> float:
        """Assess how fresh/recent the content is."""
        metadata = citation.source_metadata
        
        if not metadata.publication_date:
            return 0.5  # Neutral if no date
        
        try:
            # Simple year extraction
            year_match = re.search(r'(\d{4})', metadata.publication_date)
            if year_match:
                pub_year = int(year_match.group(1))
                current_year = datetime.now().year
                age = current_year - pub_year
                
                # Fresher content gets higher scores
                if age <= 2:
                    return 1.0
                elif age <= 5:
                    return 0.8
                elif age <= 10:
                    return 0.6
                else:
                    return 0.4
        except:
            pass
        
        return 0.5  # Default
    
    async def _verify_citations(self, citations: List[Citation]) -> None:
        """Verify citation accuracy and completeness."""
        for citation in citations:
            try:
                # Basic verification checks
                verification_status = "verified"
                
                # Check if essential metadata is present
                metadata = citation.source_metadata
                if not metadata.title or not metadata.authors:
                    verification_status = "incomplete"
                
                # Check for suspicious patterns
                if self._has_suspicious_patterns(citation):
                    verification_status = "suspicious"
                
                # Update verification status
                citation.verification_status = verification_status
                
            except Exception as e:
                logger.debug(f"Citation verification failed for {citation.id}: {e}")
                citation.verification_status = "error"
    
    def _has_suspicious_patterns(self, citation: Citation) -> bool:
        """Check for suspicious patterns in citation."""
        # Check for very short or very long titles
        title = citation.source_metadata.title or ""
        if len(title) < 5 or len(title) > 300:
            return True
        
        # Check for suspicious author patterns
        authors = citation.source_metadata.authors
        if authors:
            for author in authors:
                if len(author) < 3 or not re.match(r'^[A-Za-z\s\.\-]+$', author):
                    return True
        
        return False
    
    async def _format_citation(self, citation: Citation, style: CitationStyle) -> str:
        """Format a complete citation entry."""
        # This may be different from citation_text for bibliography entries
        return citation.citation_text
    
    async def _group_citations_by_topic(self, citations: List[Citation]) -> List[CitationGroup]:
        """Group citations by topic using simple keyword clustering."""
        groups = []
        
        # Simple grouping by source type and keywords
        source_type_groups = {}
        for citation in citations:
            source_type = citation.source_metadata.source_type
            if source_type not in source_type_groups:
                source_type_groups[source_type] = []
            source_type_groups[source_type].append(citation)
        
        for source_type, type_citations in source_type_groups.items():
            if type_citations:
                group = CitationGroup(
                    topic=source_type.value.replace("_", " ").title(),
                    citations=type_citations,
                    group_confidence=sum(c.confidence_score for c in type_citations) / len(type_citations),
                    consensus_level=0.8  # Placeholder
                )
                groups.append(group)
        
        return groups
    
    def _calculate_bibliography_statistics(self, citations: List[Citation]) -> Dict[str, Any]:
        """Calculate statistics about the bibliography."""
        if not citations:
            return {}
        
        # Source type distribution
        source_types = {}
        for citation in citations:
            source_type = citation.source_metadata.source_type.value
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        # Quality statistics
        quality_scores = [c.quality_score for c in citations]
        relevance_scores = [c.relevance_score for c in citations]
        
        # Publication year distribution
        years = []
        for citation in citations:
            pub_date = citation.source_metadata.publication_date
            if pub_date:
                year_match = re.search(r'(\d{4})', pub_date)
                if year_match:
                    years.append(int(year_match.group(1)))
        
        statistics = {
            "total_citations": len(citations),
            "source_type_distribution": source_types,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "average_relevance_score": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "publication_year_range": {
                "earliest": min(years) if years else None,
                "latest": max(years) if years else None
            },
            "verified_citations": len([c for c in citations if c.verification_status == "verified"]),
            "suspicious_citations": len([c for c in citations if c.verification_status == "suspicious"])
        }
        
        return statistics
    
    # Placeholder methods for source extractors (can be enhanced with real implementations)
    
    async def _extract_doi_metadata(self, source: str, metadata: SourceMetadata) -> Optional[SourceMetadata]:
        """Extract metadata from DOI (placeholder)."""
        doi_pattern = r'10\.\d+/[^\s]+'
        match = re.search(doi_pattern, source)
        if match:
            metadata.doi = match.group(0)
            metadata.source_type = SourceType.JOURNAL_ARTICLE
        return metadata
    
    async def _extract_isbn_metadata(self, source: str, metadata: SourceMetadata) -> Optional[SourceMetadata]:
        """Extract metadata from ISBN (placeholder)."""
        isbn_pattern = r'(?:ISBN[:\-]?\s*)?(?:978[:\-]?\s*)?(?:\d[:\-]?\s*){9}\d'
        match = re.search(isbn_pattern, source, re.IGNORECASE)
        if match:
            metadata.isbn = re.sub(r'[:\-\s]', '', match.group(0))
            metadata.source_type = SourceType.BOOK
        return metadata
    
    async def _extract_url_metadata(self, source: str, metadata: SourceMetadata) -> Optional[SourceMetadata]:
        """Extract metadata from URL."""
        if source.startswith(('http://', 'https://')):
            metadata.url = source
            
            # Try to determine source type from URL
            parsed_url = urlparse(source)
            domain = parsed_url.netloc.lower()
            
            if any(academic in domain for academic in ['arxiv', 'pubmed', 'doi', 'researchgate']):
                metadata.source_type = SourceType.ACADEMIC_PAPER
            elif any(news in domain for news in ['news', 'reuters', 'bbc', 'cnn']):
                metadata.source_type = SourceType.WEBSITE
            else:
                metadata.source_type = SourceType.WEBSITE
        
        return metadata
    
    async def _extract_text_metadata(self, source: str, metadata: SourceMetadata) -> Optional[SourceMetadata]:
        """Extract metadata from plain text source description."""
        # Simple text parsing for common patterns
        if not metadata.title and len(source) > 10:
            metadata.title = source[:100]  # Use first part as title
        
        return metadata
    
    async def _extract_filename_metadata(self, source: str, metadata: SourceMetadata) -> Optional[SourceMetadata]:
        """Extract metadata from filename."""
        if '.' in source and not source.startswith('http'):
            # Likely a filename
            filename = source.split('/')[-1]  # Get just the filename
            name_without_ext = filename.rsplit('.', 1)[0]
            
            if not metadata.title:
                metadata.title = name_without_ext.replace('_', ' ').replace('-', ' ')
            
            # Determine type by extension
            extension = filename.split('.')[-1].lower()
            if extension in ['pdf', 'doc', 'docx']:
                if 'paper' in filename.lower() or 'article' in filename.lower():
                    metadata.source_type = SourceType.ACADEMIC_PAPER
                else:
                    metadata.source_type = SourceType.REPORT
        
        return metadata
