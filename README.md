# Daraz Product Review Analyzer

A FastAPI-based tool that analyzes customer reviews from Daraz.pk products. Extracts sentiment and identifies common issues from product reviews.

## Features
- 🚀 Scrape reviews from any Daraz.pk product URL
- 🤖 AI-powered sentiment analysis using HuggingFace transformers
- 📊 Identifies common complaints in negative reviews
- ⚡ FastAPI backend with clean REST API
- 📦 Lightweight with minimal dependencies

## How It Works
1. Provide Daraz product URL
2. System fetches reviews through API
3. Analyzes sentiment (Positive/Negative/Neutral)
4. Identifies frequent issues in negative reviews
5. Returns structured JSON analysis

## Tech Stack
- Python 3
- FastAPI
- HuggingFace Transformers
- Requests

## API Endpoint
`GET /analyze?url={daraz-url}&max_pages={1-3}`

Perfect for e-commerce analytics, product research, and customer feedback analysis!
