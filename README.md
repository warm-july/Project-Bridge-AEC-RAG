# Project-Bridge-AEC-RAG

## Project Background

This project originated as part of the **Google Gen AI Intensive Course (March 31 - April 4, 2025)** hosted on Kaggle. The challenge: apply core Generative AI concepts — including document/image understanding, Retrieval-Augmented Generation (RAG), structured outputs, and function calling — to tackle a real-world problem.

Our focus: addressing the **data fragmentation crisis** in the Architecture, Engineering, and Construction (AEC) industry by building a **smart information retrieval system** tailored to AEC needs.

🔗 [Public Kaggle Notebook (Phase 1)](https://www.kaggle.com/code/junwangzero/google-gen-ai-project-capstone-2025-jw-to)

🔗 [YouTube Video Overview](https://www.youtube.com/watch?v=iwlgeVLrNbU&themeRefresh=1)

*Acknowledgement: Google's Gemini 2.5 Pro model provided valuable assistance in drafting and refining this documentation.*

---

## Problem Statement

The AEC industry, projected to surpass **$10 trillion by 2030** (Oxford Economics, 2021), remains one of the least digitized sectors. Key challenges include:

- Highly **fragmented project value chains**
- **Transient project teams**
- **Chronically low R&D spending** (McKinsey, 2019)
- **96% of project data remains unused** (FMI Corporation, 2018)
- **13% of professional time** is spent searching for project information, costing an estimated **$88 billion annually** in avoidable rework (Autodesk & FMI, 2021)

Despite producing vast amounts of data (e.g., BIM models, RFIs, job-site photos), AEC firms struggle to make knowledge easily accessible when and where it matters most.

---

## Solution Overview

We built a **Retrieval-Augmented Generation (RAG) system** that:

- **Ingests** text and visual project knowledge
- **Embeds** and stores content in **semantic vector databases** (ChromaDB)
- **Retrieves** relevant data based on user queries
- **Generates** grounded, structured responses using Google's **Gemini 2.0 Flash** model
- Supports **multimodal retrieval** — both text **and** images
- Uses **function calling** to dynamically route queries to appropriate retrieval or generation tools

The proof-of-concept shows how AEC professionals can:

- Find precedents or design details during meetings
- Access cost metrics or space planning data instantly
- Draft proposals and reports faster
- Empower new team members with self-service knowledge access

---

## Key Technologies

| Layer                  | Tools/Technologies                                                    | Purpose                                                                                   |
| ---------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **LLMs & Vision Models** | Gemini-2.0-Flash                                                      | Natural language understanding, vision-based image captioning, function calling          |
| **Vector Database**    | [ChromaDB](https://www.trychroma.com)                                 | Semantic storage and fast retrieval of embedded text and images                          |
| **Data Ingestion**      | Python, Pandas                                                        | Extracting, cleaning, enriching text and visual data                                      |
| **Embedding Models**    | Gemini text-embedding-004                                              | High-dimensional semantic embeddings for text and images                                 |
| **Retrieval Layer**     | Custom Python Functions + Gemini Function Calling                    | Query routing to text search, image search, structured response generation, image creation |
| **Output Rendering**    | JSON ➔ Markdown                                                       | Clean, structured display of citations and media                                          |
| **Development Platform** | [Kaggle Notebooks](https://kaggle.com/)                               | Interactive development, demonstration, and public access                                |

---

## Phase 1: First Prototype (Completed)

The Phase 1 implementation includes:

- A working multimodal RAG system demonstrated on **AEC thought leadership content** (using the [Gensler Research Library](https://www.gensler.com/research-library))
- Integration of AI-based image captioning and OCR for richer visual context
- Function-calling powered intelligent retrieval workflows
- Examples of article generation via few-shot prompting
- Structured JSON outputs for consistent formatting and citation control

Explore it here: 

🔗 [Google Gen AI Project Capstone - Jun Wang (Kaggle Notebook)](https://www.kaggle.com/code/junwangzero/google-gen-ai-project-capstone-2025-jw-to)


---

## Phase 2: Future Development (Planned)

To make the system production-ready, future improvements include:

### Data, Models & Retrieval
- Robust ingestion of diverse formats: PDFs, BIM summaries, emails
- Specialized embeddings for documents, diagrams, and structured data
- Multi-modal fusion retrieval combining text and image results

### Intelligence & Interface
- Fine-tuned chunking for large documents
- Multi-step agentic workflows for complex queries
- Live data access via secure API integration

### UX & Deployment
- Custom web/app interface for real-world use
- Faceted search, interactive previews, visual-first layout
- Advanced structured output control for better media/citation formatting

---

## References

- Autodesk & FMI. (2021). [Harnessing the Data Advantage in Construction](https://construction.autodesk.com/resources/guides/harnessing-data-advantage-in-construction/)
- FMI Corporation. (2018). [Big Data = Big Questions](https://fmicorp.com/uploads/media/FMI_BigDataReport.pdf)
- McKinsey & Company. (2019). [Decoding Digital Transformation in Construction](https://www.mckinsey.com/industries/engineering-construction-and-building-materials/our-insights/decoding-digital-transformation-in-construction)
- Oxford Economics. (2021). [Future of Construction: A Global Forecast to 2030](https://www.oxfordeconomics.com/wp-content/uploads/2023/08/Future-of-Construction-Full-Report.pdf)

---

## License

This project is currently intended for **educational demonstration purposes only**. Future licensing considerations will be addressed with Phase 2.

---

> ✨ *By connecting fragmented insights to real-time needs, AI can help AEC professionals design and build better.*


[GitHub Page](https://warm-july.github.io/Project-Bridge-AEC-RAG/)
