# MongoDB Chat Agent - Data Flow Analysis

## Overview
The **MongoDB Chat Agent** allows users to query their database using natural language. It employs a hybrid approach: **Template-based execution** for common queries (fast path) and **LLM-based generation** for complex queries (smart path).

## High-Level Flowchart

```mermaid
graph TD
    User([User Input]) --> SafetyCheck{Safety Check}
    SafetyCheck -- "Has delete/drop" --> Blocked[Return Error]
    SafetyCheck -- "Safe" --> TemplateCheck{Match Template?}

    %% Path 1: Template (Fast Path)
    TemplateCheck -- "Yes (e.g. Orders + Customer)" --> BuildTemplate[Build Aggregation Pipeline]
    BuildTemplate --> ExecuteDB

    %% Path 2: LLM (Smart Path)
    TemplateCheck -- "No" --> ContextEmbed[Vector Search Logic]
    ContextEmbed -->|Embed Question| VectorDB[(Embeddings DB)]
    VectorDB -->|Return Relevant Collections| PromptBuilder[Build Schema Context]
    PromptBuilder -->|System + User Prompt| LLM[Local LLM (Qwen2.5)]
    LLM -->|Generate Code| Parser[Query Parser & Fixer]
    Parser --> ExecuteDB[(MongoDB Execution)]
    
    %% Execution & Result
    ExecuteDB -->|Raw Results| Formatter[Response Formatter]
    Formatter --> FinalOutput([Final Response])

    %% Error Handling
    Parser -- "Error/Invlaid" --> ErrorHandler[Error Message]
    ErrorHandler --> FinalOutput
```

## detailed Data Flow Steps

### 1. User Input & Validation
*   **Input**: User types a natural language question (e.g., "Show me top 5 customers").
*   **Safety**: System checks for prohibited keywords (`drop`, `delete`, `update`, `insert`) to ensure read-only access.

### 2. Smart Routing (Hybrid Logic)
The system decides how to handle the query:
*   **Template Path**: Fast regular-expression checks detect common "join" patterns (e.g., asking for orders *and* customer details). If matched, it constructs a pre-defined `$lookup` pipeline immediately.
*   **LLM Path**: If no template matches, it proceeds to the AI generation flow.

### 3. Context Retrieval (RAG - Retrieval Augmented Generation)
*   **Vector Search**: The user's question is converted into a vector embedding using `SentenceTransformer`.
*   **Schema Matching**: The system searches `embeddings.pkl` to find the most relevant database collections (e.g., matching "sales" to the `orders` collection).
*   **Context Building**: A text prompt is constructed that includes *only* the schema (field names, types) of the relevant collections to reduce LLM hallucination.

### 4. LLM Query Generation
*   **Model**: The prompt is sent to a local LLM (Ollama/Qwen2.5-3b).
*   **Instruction**: The System Prompt explicitly enforces rules (valid JSON, no new fields, correct `$lookup` syntax).
*   **Output**: The LLM returns a raw MongoDB query string (e.g., `db.orders.aggregate[...]`).

### 5. Parsing & Execution
*   **Sanitization**: The system cleans the LLM output (removes markdown, fixes quote issues, balances brackets).
*   **Auto-Correction**: It detects logical errors, such as swapping `$match` and `$unwind` stages if they are in the wrong order.
*   **Execution**: The final query is run against the MongoDB database `ai_test_db`.

### 6. Response Formatting
*   **Formatting**: Raw JSON results are converted into a readable list or summary.
*   **Display**: Special handling exists for single-aggregation results (like "Total Sales: $500") vs. sets of documents.

## Key Files
*   **`mongo_chat_agent.py`**: The main controller containing the `MongoDBChatAgent` class.
*   **`llm_integration.py`**: Handles communication with the local Ollama instance.
*   **`mongodb_schema_embedding_system.py`**: Utility to analyze the database and create the `embeddings.pkl` file used for context retrieval.
