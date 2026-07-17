import './ExampleChips.css';

interface ExampleChipsProps {
  onSelect: (query: string) => void;
  disabled: boolean;
}

const EXAMPLES = [
  "What is the effective concentration of spores?",
  "How is the fungus formulated for field use?",
  "What are the target pests for this biopesticide?"
];

export function ExampleChips({ onSelect, disabled }: ExampleChipsProps) {
  return (
    <div className="chips-container">
      <span className="chips-label">Try asking:</span>
      <div className="chips-list">
        {EXAMPLES.map((example, index) => (
          <button
            key={index}
            className="chip-button"
            onClick={() => onSelect(example)}
            disabled={disabled}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
