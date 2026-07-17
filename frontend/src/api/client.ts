export interface Citation {
  paragraph_id: string;
  section: string;
  quote: string;
  score: number;
}

export interface QueryResponse {
  answer: string;
  confidence: 'high' | 'partial' | 'not_found';
  citations: Citation[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function postQuestion(path: string, question: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  });

  if (!response.ok) {
    let detail = 'Failed to query the backend';
    try {
      const errorBody = await response.json();
      detail = errorBody.detail ?? detail;
    } catch {
      // Keep the generic message when the backend returns a non-JSON error.
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function queryEvidence(question: string): Promise<QueryResponse> {
  return postQuestion('/query/evidence', question);
}

export async function queryBackend(question: string): Promise<QueryResponse> {
  return postQuestion('/query', question);
}
