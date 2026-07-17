import { useMemo, useState } from 'react';
import { FileText, ShieldCheck, ShieldX } from 'lucide-react';
import type { Citation, QueryResponse } from '../api/client';
import { CitationPanel } from './CitationPanel';
import './AnswerCard.css';

interface AnswerCardProps {
  response: QueryResponse | null;
  isLoading: boolean;
  isUpdating?: boolean;
}

export function AnswerCard({ response, isLoading, isUpdating = false }: AnswerCardProps) {
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  const citationsById = useMemo(() => {
    const byId = new Map<string, Citation>();
    response?.citations.forEach((citation) => {
      byId.set(citation.paragraph_id, citation);
    });
    return byId;
  }, [response]);

  if (isLoading) {
    return (
      <div className="answer-card loading-skeleton animate-pulse">
        <div className="skeleton-line" style={{ width: '80%' }}></div>
        <div className="skeleton-line" style={{ width: '60%' }}></div>
        <div className="skeleton-line" style={{ width: '70%' }}></div>
      </div>
    );
  }

  if (!response) {
    return null;
  }

  const activeCitation = activeCitationId ? citationsById.get(activeCitationId) : null;

  const getConfidenceBadge = () => {
    switch (response.confidence) {
      case 'high':
        return (
          <div className="confidence-badge high">
            <ShieldCheck size={16} />
            <span>High Confidence</span>
          </div>
        );
      case 'partial':
        return null;
      case 'not_found':
        return (
          <div className="confidence-badge not-found">
            <ShieldX size={16} />
            <span>Not Found</span>
          </div>
        );
      default:
        return null;
    }
  };

  const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const renderAnswerText = (text: string) => {
    const citationIds = response.citations
      .map((citation) => citation.paragraph_id)
      .filter(Boolean)
      .sort((a, b) => b.length - a.length);

    if (citationIds.length === 0) {
      return text;
    }

    const citationPattern = new RegExp(`(${citationIds.map(escapeRegex).join('|')})`, 'g');
    const parts = text.split(citationPattern);

    return parts.map((part, i) => {
      if (citationsById.has(part)) {
        return (
          <button
            key={`${part}-${i}`}
            type="button"
            className={`inline-citation ${activeCitationId === part ? 'active' : ''}`}
            onClick={() => setActiveCitationId(activeCitationId === part ? null : part)}
            aria-pressed={activeCitationId === part}
          >
            {part}
          </button>
        );
      }
      return <span key={i}>{part}</span>;
    });
  };

  const previewCitations = response.answer
    .split(/\s+/)
    .some((word) => response.citations.some((citation) => word.includes(citation.paragraph_id)))
    ? []
    : response.citations;

  return (
    <div className="answer-card animate-fade-in">
      <div className="answer-header">
        <h3 className="answer-title">Synthesized Answer</h3>
        {getConfidenceBadge()}
      </div>

      {isUpdating && (
        <div className="answer-updating" role="status">
          Retrieved evidence is shown. Final cited answer is still being synthesized.
        </div>
      )}
      
      <div className="answer-text">
        {renderAnswerText(response.answer)}
      </div>

      {activeCitation && (
        <div className="source-peek animate-fade-in">
          <div className="source-peek-header">
            <FileText size={16} />
            <span className="source-peek-label">{activeCitation.paragraph_id}</span>
            <span className="source-peek-section">{activeCitation.section}</span>
          </div>
          <blockquote>{activeCitation.quote}</blockquote>
          <div className="source-peek-score">
            Evidence score {(activeCitation.score * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {previewCitations.length > 0 && (
        <div className="inline-source-strip" aria-label="Retrieved source locators">
          {previewCitations.map((citation) => (
            <button
              key={citation.paragraph_id}
              type="button"
              className={`inline-citation ${activeCitationId === citation.paragraph_id ? 'active' : ''}`}
              onClick={() =>
                setActiveCitationId(
                  activeCitationId === citation.paragraph_id ? null : citation.paragraph_id
                )
              }
              aria-pressed={activeCitationId === citation.paragraph_id}
            >
              {citation.paragraph_id}
            </button>
          ))}
        </div>
      )}

      {response.citations.length > 0 && (
        <div className="citations-section">
          <h4 className="citations-title">Sources &amp; Citations</h4>
          <div className="citations-list">
            {response.citations.map((citation, index) => (
              <CitationPanel
                key={`${citation.paragraph_id}-${index}`}
                citation={citation}
                index={index}
                isOpen={activeCitationId === citation.paragraph_id}
                highlighted={activeCitationId === citation.paragraph_id}
                onToggle={() =>
                  setActiveCitationId(
                    activeCitationId === citation.paragraph_id ? null : citation.paragraph_id
                  )
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
