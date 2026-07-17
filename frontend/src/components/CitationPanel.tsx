import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import type { Citation } from '../api/client';
import './CitationPanel.css';

interface CitationPanelProps {
  citation: Citation;
  index: number;
  isOpen?: boolean;
  highlighted?: boolean;
  onToggle?: () => void;
}

export function CitationPanel({
  citation,
  index,
  isOpen,
  highlighted = false,
  onToggle
}: CitationPanelProps) {
  const [localExpanded, setLocalExpanded] = useState(false);
  const isExpanded = isOpen ?? localExpanded;
  const toggleExpanded = onToggle ?? (() => setLocalExpanded(!localExpanded));
  const locator = citation.paragraph_id;

  return (
    <div className={`citation-container ${highlighted ? 'highlighted' : ''}`}>
      <button
        className="citation-toggle"
        onClick={toggleExpanded}
        aria-expanded={isExpanded}
      >
        <span className="citation-marker">{locator}</span>
        <span className="citation-preview">Source {index + 1}</span>
        {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>

      {isExpanded && (
        <div className="citation-content animate-fade-in">
          <div className="citation-header">
            <FileText size={14} />
            <span className="citation-doc-title">Retrieved patent excerpt</span>
            <span className="citation-section">&bull; {citation.section}</span>
          </div>
          <div className="citation-quote">
            "{citation.quote}"
          </div>
          <div className="citation-footer">
            <span>Score: {(citation.score * 100).toFixed(1)}% match</span>
            <span>{locator}</span>
          </div>
        </div>
      )}
    </div>
  );
}
