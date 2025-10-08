// BAML clients configuration

// Default client configuration - will be overridden at runtime
client<llm> "openai/gpt-4o-mini" {
  provider "openai"
  options {
    model "gpt-4o-mini"
    api_key env.OPENAI_API_KEY
  }
}

// Gemini client configuration
client<llm> "google/gemini-2.0-flash" {
  provider "google-ai"
  options {
    model "gemini-2.0-flash"
    api_key env.GEMINI_API_KEY
  }
}

client<llm> "google/gemini-2.5-flash" {
  provider "google-ai"
  options {
    model "gemini-2.5-flash"
    api_key env.GEMINI_API_KEY
  }
}

