import { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import './QueryBox.css';

interface QueryBoxProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export function QueryBox({ onSearch, isLoading }: QueryBoxProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query.trim());
    }
  };

  return (
    <form className="query-box-container" onSubmit={handleSubmit}>
      <div className="query-input-wrapper">
        <Search className="query-icon" size={20} />
        <input
          type="text"
          className="query-input"
          placeholder="Ask a question about the patent..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          className={`query-submit ${query.trim() ? 'active' : ''}`}
          disabled={!query.trim() || isLoading}
        >
          {isLoading ? <Loader2 className="spinner" size={18} /> : 'Ask'}
        </button>
      </div>
    </form>
  );
}
