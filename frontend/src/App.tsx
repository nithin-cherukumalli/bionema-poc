import { useRef, useState } from 'react';
import { Leaf } from 'lucide-react';
import { QueryBox } from './components/QueryBox';
import { ExampleChips } from './components/ExampleChips';
import { AnswerCard } from './components/AnswerCard';
import type { QueryResponse } from './api/client';
import { queryBackend, queryEvidence } from './api/client';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeRequestId = useRef(0);

  const handleSearch = async (query: string) => {
    const requestId = activeRequestId.current + 1;
    activeRequestId.current = requestId;
    setIsLoading(true);
    setIsSynthesizing(false);
    setError(null);
    setResponse(null);

    try {
      const evidence = await queryEvidence(query);
      if (activeRequestId.current !== requestId) {
        return;
      }

      setResponse(evidence);
      setIsLoading(false);
      setIsSynthesizing(true);

      const finalAnswer = await queryBackend(query);
      if (activeRequestId.current !== requestId) {
        return;
      }

      setResponse(finalAnswer);
    } catch (err) {
      if (activeRequestId.current === requestId) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
      }
    } finally {
      if (activeRequestId.current === requestId) {
        setIsLoading(false);
        setIsSynthesizing(false);
      }
    }
  };

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="logo-container">
          <Leaf className="logo-icon" size={28} />
          <h1 className="logo-text">Bionema <span className="logo-highlight">Retrieval</span></h1>
        </div>
        <p className="app-subtitle">
          Query technical documents with retrieved evidence and citable source locators.
        </p>
      </header>

      <main className="app-main">
        <QueryBox onSearch={handleSearch} isLoading={isLoading} />
        
        {!response && !isLoading && (
          <ExampleChips onSelect={handleSearch} disabled={isLoading} />
        )}

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {(isLoading || response) && (
          <AnswerCard
            response={response}
            isLoading={isLoading && !response}
            isUpdating={isSynthesizing}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>Bionema POC &bull; Retrieved answers with inline and expandable citations</p>
      </footer>
    </div>
  );
}

export default App;
