from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Data Assistant: Chat with Database Architecture', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# --- CONTENT GENERATION ---

intro = """
This document outlines the technical architecture of the "Chat with Database" feature.
The system leverages a Local LLM (Qwen 2.5:3b) combined with a Python-based "Hybrid Router" to deliver low-latency, high-accuracy database interactions.

Current State:
- Database: MongoDB (Local)
- LLM: Qwen 2.5:3b (via Ollama)
- Framework: Streamlit + Python
"""

section_flow = """
The system follows a 4-Stage Pipeline for every user query:

1. INTENT RECOGNITION (Python Router)
   - Input: User natural language ("Show top selling products").
   - Logic: Regex & Keyword Matching.
   - Action: 
     - If "Hi/Hello": Return Static Greeting (Instant).
     - If "Sales/Orders": Select 'orders' collection schema.
     - If Complex: Fallback to Full Schema.
   - Benefit: Reduces token usage and skips LLM for trivial tasks.

2. QUERY GENERATION (LLM Pass 1)
   - Input: User Question + JSON Schema of selected collection.
   - Prompt Engineering: 
     - Few-Shot Examples (Compare Years, Top N).
     - Strict Rules (e.g. "Use $lookup for names", "Map 'selling' to orders").
   - Output: Valid MongoDB Aggregation Pipeline (JSON).

3. EXECUTION DETECTOR (Python Controller)
   - Logic: Parses LLM JSON output.
   - Safety: Checks for destructive commands (drop/delete).
   - Action: Executes pipeline on MongoDB using PyMongo.
   - Handling: Catches empty results and provides fallback/errors.

4. ANSWER SYNTHESIS (LLM Pass 2)
   - Input: Raw JSON/DataFrame results from Database.
   - Logic: "Convert this data into a helpful answer".
   - Output: Natural Language Narrative ("The top product is...").
   - Result: Combined Table  + Narrative displayed to user.
"""

section_roles = """
ROLE OF LLM (The Brain):
- Translation: Converts English -> MongoDB Query Language (MQL).
- Reasoning: Decides HOW to group/sort/filter based on schema.
- Synthesis:Reads technical data and summarizes it for humans.
- Limitations: Stochastic (randomness). Mitigated by strict Python guardrails.

ROLE OF PYTHON (The Nervous System):
- Context Management: Decides WHICH schema to show the LLM (Router).
- Execution: physically runs the query.
- Guardrails: Prevents crashing on bad JSON, handles "No Data" states.
- Formatting: Renders UI (Charts/Tables).
"""

section_prompt = """
PROMPT ENGINEERING STRATEGY:
The system uses a "Chain of Thought" style system prompt.

Key Elements:
1. Role Definition: "You are a MongoDB Expert."
2. Schema Injection: Dynamically inserted based on Router.
3. Logical Rules:
   - "For 'Top 10', sort by amount descending."
   - "If ID field present, usage $lookup."
   - "Unless specified, query ALL TIME (do not limit to 2026)."
4. Golden Examples: Pre-written Q&A pairs for complex logic (e.g. Year-over-Year comparison).
"""

pdf.chapter_title("1. Overview")
pdf.chapter_body(intro)

pdf.chapter_title("2. Full Request Flow")
pdf.chapter_body(section_flow)

pdf.chapter_title("3. Component Roles")
pdf.chapter_body(section_roles)

pdf.chapter_title("4. Prompt Strategy")
pdf.chapter_body(section_prompt)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"d:/ai-data_assistance/Chat_Database_Architecture_{timestamp}.pdf"
pdf.output(filename, 'F')
print(f"PDF Generated: {filename}")
