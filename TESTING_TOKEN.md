# LLM Testing Token

One-click login for LLM agents without exposing API keys in the UI.

## Setup

Add `TESTING_TOKEN` to your environment:

### Option 1: .env file (recommended)
```bash
# .env
TESTING_TOKEN=mb_api_your_token_here
```

### Option 2: System environment
```bash
export TESTING_TOKEN=mb_api_your_token_here
```

## Usage

When `TESTING_TOKEN` is detected, the login page shows:
- Green **"Login with Testing Token"** button (one-click login)
- Manual login option below

Click the green button to auto-login without entering credentials.

## Why?

- LLM agents can authenticate without seeing/exposing tokens
- No need to paste API keys into form fields
- Perfect for automated testing and agent workflows
