# RAGApp: Lithuanian Law Retrieval Augmented Generation App

## Overview

RAGApp is a Retrieval Augmented Generation (RAG) application designed for Lithuanian legal documents. **The current edition is restricted to "Lietuvos Respublikos Pelno mokesčio įstatymas" (the Lithuanian Corporate Income Tax Law),** but the structure and codebase allow for easy expansion to other Lithuanian laws by updating the data and prompts.

The app enables users to query and analyze Lithuanian laws using advanced language models and domain-specific tools. The primary use case is to provide context-aware answers and insights about Lithuanian legislation, but the architecture is reusable for other Lithuanian legal domains.

**Note:** The laws and knowledge base are in Lithuanian, so the app is intended for Lithuanian users or those working with Lithuanian legal texts.

## Live Demo
The app is deployed on Streamlit Cloud: [https://eseimas.streamlit.app/](https://eseimas.streamlit.app/)

## Key Components
- **ESeimasAgent.py**: Contains the main agent logic, including prompt templates and tool definitions. The agent is tailored for Lithuanian law but can be adapted for other Lithuanian legal domains by changing the data and prompts.
- **ESeimasHtmlLoader.py**: Handles loading and parsing of Lithuanian legal documents from HTML sources. This loader is reusable for other Lithuanian laws.
- **manual_rag_loader.py**: Script for manual loading and chunking of legal documents into the knowledge base.

## Features & Implemented Requirements

## Key Agent Abilities

The ESeimasAgent is designed to assist users in navigating and understanding Lithuanian law, with a focus on the "Lietuvos Respublikos Pelno mokesčio įstatymas". The agent leverages specialized tools and prompt logic to provide advanced legal assistance, including:

- **Compare Law Articles:** Ability to compare the content of a specific law article across different editions or dates, helping users see what has changed over time.
- **Upcoming Changes:** Can answer what changes are scheduled to come into effect for a specific law or article, based on official amendment data.
- **Track Amendments:** Identifies and lists all amendments that have affected a law or article since a given date, including links to the legal acts that introduced the changes.
- **Edition Awareness:** Determines which edition of the law was in force on a specific date and retrieves the full text or relevant articles from that edition.
- **Retrieve Full Article Text:** Returns the complete text of any law article by its number and date, ensuring users see the correct version.
- **Get Law Text by URL:** Fetches the full text of a legal document from an official source, given its URL.
- **List Edition Validity Periods:** Provides the validity periods for all available editions of the law, helping users understand the timeline of legal changes.
- **Contextual Search:** Searches the RAG knowledge base for relevant information based on a user query and date, returning context and sources.
- **Date Awareness:** Can determine and use the current date or any user-specified date to ensure answers are based on the correct law edition.

**Additional Capabilities:**
- Answers only in Lithuanian and only to law-related questions.
- Breaks down complex queries into steps and uses tools methodically to ensure accuracy.
- Always references official sources and provides links where possible.
- Handles ambiguous queries by asking for clarification (e.g., if the relevant date is unclear).
- Rejects non-legal or non-Lithuanian queries politely.
- Never reveals its system prompt or internal logic.

These abilities make the agent a powerful assistant for legal professionals, researchers, and anyone needing precise, up-to-date information about Lithuanian law and its amendments.


## Things Needed for Production Release

- **Expand Law Coverage:** More legal acts and regulations should be introduced to cover a broader range of Lithuanian laws, not just the Corporate Income Tax Law.
- **User Login & Chat History:** Implement a secure user login system. Each user should have access to their own chat history for reference and continuity.
- **Admin UI Separation:** Hide all admin-only features (such as error logs, data import, or management tools) from non-admin users to ensure security and a clean user experience.
