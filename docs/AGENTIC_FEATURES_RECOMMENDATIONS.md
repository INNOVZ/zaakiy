# Agentic Features Recommendations for ZaaKy AI Platform

## Overview
This document outlines practical agentic features that can be incrementally integrated into your existing RAG system without major architectural changes.

## Current Architecture Strengths
- ✅ Modular service design (ChatService, DocumentRetrievalService, ResponseGenerationService)
- ✅ Intent detection system
- ✅ Query enhancement capabilities
- ✅ Multiple retrieval strategies (semantic, hybrid, keyword)
- ✅ Web scraping infrastructure
- ✅ Conversation management

## Recommended Agentic Features (Priority Order)

### 1. Adaptive Retrieval Loop ⭐⭐⭐ (HIGHEST PRIORITY)
**Impact**: High | **Complexity**: Medium | **Effort**: 2-3 days

**What it does**: Automatically refines retrieval if initial results are insufficient.

**Implementation**:
```python
# Add to DocumentRetrievalService
async def retrieve_with_validation(
    self,
    queries: List[str],
    original_message: str,
    max_iterations: int = 2
) -> Tuple[List[Dict], bool]:
    """Retrieve documents with quality validation"""
    for iteration in range(max_iterations):
        documents = await self.retrieve_documents(queries)

        # Validate quality using LLM
        is_sufficient = await self._validate_retrieval_quality(
            original_message, documents
        )

        if is_sufficient or iteration == max_iterations - 1:
            return documents, is_sufficient

        # Refine query and retry
        queries = await self._refine_queries(original_message, documents, queries)

    return documents, False
```

**Benefits**:
- Improves answer quality without user intervention
- Handles ambiguous queries better
- Reduces "I don't know" responses

---

### 2. Query Decomposition ⭐⭐⭐ (HIGH PRIORITY)
**Impact**: High | **Complexity**: Medium | **Effort**: 3-4 days

**What it does**: Breaks complex multi-part questions into sub-queries, retrieves for each, then synthesizes.

**Example**:
- User: "Compare Product A and B, then tell me pricing and shipping options"
- Decomposed:
  1. "Product A features and specifications"
  2. "Product B features and specifications"
  3. "Product A pricing"
  4. "Product B pricing"
  5. "Shipping options for both products"

**Implementation**:
```python
# Add to ResponseGenerationService
async def decompose_complex_query(
    self,
    message: str,
    intent_result: IntentResult
) -> List[str]:
    """Break complex queries into sub-queries"""
    if intent_result.primary_intent in [IntentType.COMPARISON, IntentType.RECOMMENDATION]:
        # Use LLM to decompose
        decomposition_prompt = f"""
        Break this query into specific sub-queries that can be answered independently:
        Query: {message}

        Return JSON array of sub-queries.
        """
        # Call LLM and parse sub-queries
        return parsed_sub_queries

    return [message]  # Simple query, no decomposition needed
```

**Benefits**:
- Handles complex questions that need multiple data points
- Better coverage of user intent
- More accurate comparisons and recommendations

---

### 3. Confidence-Based Actions ⭐⭐ (MEDIUM PRIORITY)
**Impact**: Medium | **Complexity**: Low | **Effort**: 1-2 days

**What it does**: If response confidence is low, either ask clarifying questions or retrieve more documents.

**Implementation**:
```python
# Add to ResponseGenerationService
async def generate_with_confidence_check(
    self,
    message: str,
    retrieved_documents: List[Dict],
    response: str
) -> Dict[str, Any]:
    """Generate response and check confidence"""
    # Get confidence score from LLM
    confidence = await self._assess_response_confidence(
        message, retrieved_documents, response
    )

    if confidence < 0.6:  # Low confidence threshold
        # Option 1: Ask clarifying question
        if len(retrieved_documents) < 3:
            clarifying_question = await self._generate_clarifying_question(message)
            return {
                "response": clarifying_question,
                "confidence": confidence,
                "action": "clarify"
            }

        # Option 2: Retrieve more documents
        additional_docs = await self.document_retrieval.retrieve_documents(
            queries=[message],
            intent_config={"k_values": {"initial": 10}}
        )
        # Regenerate with more context
        return await self.generate_enhanced_response(...)

    return {"response": response, "confidence": confidence, "action": "answer"}
```

**Benefits**:
- Reduces incorrect answers
- Improves user experience with proactive clarification
- Better handling of edge cases

---

### 4. Tool-Based Actions ⭐⭐ (MEDIUM PRIORITY)
**Impact**: Medium | **Complexity**: Medium | **Effort**: 4-5 days

**What it does**: Integrates external tools (calculator, web search, contact lookup) that the LLM can invoke when needed.

**Implementation**:
```python
# New file: backend/app/services/chat/tool_orchestrator.py
class ToolOrchestrator:
    """Manages tool execution for agentic behavior"""

    AVAILABLE_TOOLS = {
        "calculator": CalculatorTool(),
        "web_search": WebSearchTool(),  # Use your existing scraper
        "contact_lookup": ContactLookupTool(),
        "product_search": ProductSearchTool(),
    }

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return results"""
        if tool_name not in self.AVAILABLE_TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self.AVAILABLE_TOOLS[tool_name]
        return await tool.execute(parameters)

# In ResponseGenerationService, add tool calling:
async def generate_with_tools(
    self,
    message: str,
    conversation_history: List[Dict]
) -> Dict[str, Any]:
    """Generate response with tool support"""
    # Let LLM decide if tools are needed
    tool_calls = await self._detect_tool_needs(message, conversation_history)

    tool_results = {}
    for tool_call in tool_calls:
        result = await self.tool_orchestrator.execute_tool(
            tool_call["tool"],
            tool_call["parameters"]
        )
        tool_results[tool_call["tool"]] = result

    # Generate final response with tool results
    return await self._generate_final_response(message, tool_results)
```

**Benefits**:
- Handles questions requiring calculations or external data
- Better integration with your web scraping capabilities
- More dynamic responses

---

### 5. Self-Correction Mechanism ⭐ (LOWER PRIORITY)
**Impact**: Medium | **Complexity**: Medium | **Effort**: 2-3 days

**What it does**: Validates generated responses against retrieved documents and corrects inconsistencies.

**Implementation**:
```python
# Add to ResponseGenerationService
async def generate_with_self_correction(
    self,
    message: str,
    retrieved_documents: List[Dict],
    initial_response: str
) -> Dict[str, Any]:
    """Generate response with self-validation"""
    # Validate response accuracy
    validation_result = await self._validate_response_accuracy(
        message, retrieved_documents, initial_response
    )

    if validation_result["is_accurate"]:
        return {"response": initial_response, "corrections": []}

    # Generate corrected response
    corrected_response = await self._generate_corrected_response(
        message,
        retrieved_documents,
        initial_response,
        validation_result["inconsistencies"]
    )

    return {
        "response": corrected_response,
        "corrections": validation_result["inconsistencies"],
        "original_response": initial_response
    }
```

**Benefits**:
- Reduces hallucination
- Improves factual accuracy
- Better quality assurance

---

### 6. Multi-Turn Planning ⭐ (LOWER PRIORITY)
**Impact**: Low-Medium | **Complexity**: High | **Effort**: 5-7 days

**What it does**: Maintains conversation state and plans what information still needs to be retrieved.

**Implementation**:
```python
# Extend ConversationManager
class ConversationPlanner:
    """Tracks conversation goals and plans retrieval"""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.answered_topics = set()
        self.pending_topics = []

    async def plan_next_retrieval(
        self,
        current_message: str,
        conversation_history: List[Dict]
    ) -> List[str]:
        """Determine what information still needs to be retrieved"""
        # Analyze what's been answered
        # Identify gaps
        # Return prioritized retrieval queries
        pass
```

**Benefits**:
- Better multi-turn conversations
- More efficient retrieval
- Tracks conversation goals

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
1. ✅ Confidence-Based Actions (1-2 days)
2. ✅ Adaptive Retrieval Loop (2-3 days)

### Phase 2: Core Features (Week 3-4)
3. ✅ Query Decomposition (3-4 days)
4. ✅ Self-Correction Mechanism (2-3 days)

### Phase 3: Advanced Features (Week 5-6)
5. ✅ Tool-Based Actions (4-5 days)
6. ✅ Multi-Turn Planning (5-7 days)

## Integration Points

### Files to Modify:
- `backend/app/services/chat/document_retrieval_service.py` - Add adaptive retrieval
- `backend/app/services/chat/response_generation_service.py` - Add confidence, decomposition, self-correction
- `backend/app/services/chat/chat_service.py` - Orchestrate new features
- `backend/app/services/chat/conversation_manager.py` - Add planning support

### New Files to Create:
- `backend/app/services/chat/tool_orchestrator.py` - Tool management
- `backend/app/services/chat/query_decomposer.py` - Query decomposition logic
- `backend/app/services/chat/response_validator.py` - Response validation

## Configuration

Add to `chatbot_config`:
```python
{
    "agentic_features": {
        "adaptive_retrieval": {
            "enabled": true,
            "max_iterations": 2,
            "confidence_threshold": 0.7
        },
        "query_decomposition": {
            "enabled": true,
            "min_complexity_score": 0.6
        },
        "confidence_based_actions": {
            "enabled": true,
            "low_confidence_threshold": 0.6,
            "action": "clarify"  # or "retrieve_more"
        },
        "self_correction": {
            "enabled": true,
            "validation_enabled": true
        }
    }
}
```

## Testing Strategy

1. **Unit Tests**: Test each feature in isolation
2. **Integration Tests**: Test feature combinations
3. **A/B Testing**: Compare agentic vs non-agentic responses
4. **Performance Tests**: Ensure latency doesn't increase significantly

## Monitoring & Metrics

Track:
- Retrieval iteration count (adaptive retrieval)
- Query decomposition rate
- Confidence scores distribution
- Self-correction frequency
- Tool usage statistics
- Response quality improvements

## Cost Considerations

- **Adaptive Retrieval**: +20-30% token usage (worth it for quality)
- **Query Decomposition**: +50-100% token usage (only for complex queries)
- **Self-Correction**: +30-40% token usage (only when corrections needed)
- **Tool-Based Actions**: Variable (depends on tool usage)

## Conclusion

Start with **Adaptive Retrieval Loop** and **Confidence-Based Actions** - these provide the best ROI with minimal complexity. Then gradually add other features based on user feedback and performance metrics.
