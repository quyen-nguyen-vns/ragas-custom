// BAML configuration for Ragas query generation
// This ensures robust JSON parsing even with malformed LLM outputs

class GeneratedQueryAnswer {
  query string
  answer string
}

class Persona {
  name string
  role_description string
}

class QueryCondition {
  persona Persona
  term string
  query_style string
  query_length string
  context string
}

class QueryConditions {
  persona Persona
  themes string[]
  query_style string
  query_length string
  context string[]
}

// Single-hop query generation
function GenerateSingleHopQuery(input: QueryCondition) -> GeneratedQueryAnswer {
  client "openai/gpt-4o-mini"
  
  prompt #"
    Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) 
    and the provided context. Ensure the answer is entirely faithful to the context, using only the information 
    directly from the provided context.
    
    ### Instructions:
    1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question 
       that aligns with the persona's perspective and incorporates the term.
    2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer 
       to the query. Do not add any information not included in or inferable from the context.
    3. **Language Requirement**: Generate both the query and answer in English only, regardless of the input language. 
       Do not use any Thai words, terms, or phrases. Translate all technical terms to English or use generic English descriptions.
    
    ## Input:
    Persona: {{ input.persona.name }}
    Persona Role: {{ input.persona.role_description }}
    Term: {{ input.term }}
    Query Style: {{ input.query_style }}
    Query Length: {{ input.query_length }}
    Context: {{ input.context }}
    
    {{ ctx.output_format }}
  "#
}

// Multi-hop query generation
function GenerateMultiHopQuery(input: QueryConditions) -> GeneratedQueryAnswer {
  client "openai/gpt-4o-mini"
  
  prompt #"
    Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) 
    and the provided context. The themes represent a set of phrases either extracted or generated from the 
    context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query 
    explicitly incorporates these themes.
    
    ### Instructions:
    1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining 
       information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more 
       themes and reflects their relevance to the context.
    2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to 
       the query. Avoid adding information that is not directly present or inferable from the given context.
    3. **Multi-Hop Context Tags**:
       - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.
       - Ensure the query uses information from at least two segments and connects them meaningfully.
    4. **Language Requirement**: Generate both the query and answer in English only, regardless of the input language. 
       Do not use any Thai words, terms, or phrases. Translate all technical terms to English or use generic English descriptions.
    
    ## Input:
    Persona: {{ input.persona.name }}
    Persona Role: {{ input.persona.role_description }}
    Themes: {{ input.themes }}
    Query Style: {{ input.query_style }}
    Query Length: {{ input.query_length }}
    Context: {{ input.context }}
    
    {{ ctx.output_format }}
  "#
}

