const CATS = {
  thinking: { label: 'Thinking', sw: 'var(--c-think)', ink: 'var(--c-think-ink)' },
  readonly: { label: 'Read-only tool', sw: 'var(--c-read)', ink: 'var(--c-read-ink)' },
  mutating: { label: 'Mutating tool', sw: 'var(--c-mut)', ink: 'var(--c-mut-ink)' },
  planning: { label: 'Planning', sw: 'var(--c-plan)', ink: 'var(--c-plan-ink)' },
  subagent: { label: 'Subagent', sw: 'var(--c-plan)', ink: 'var(--c-plan-ink)' },
  other: { label: 'Other tool', sw: 'var(--c-other)', ink: 'var(--c-other-ink)' },
  text: { label: 'Response text', sw: 'var(--c-text)', ink: 'var(--c-text-ink)' },
  dead: { label: 'Dead end', sw: 'var(--c-dead)', ink: 'var(--c-dead-ink)' },
}
const ORDER = ['thinking', 'readonly', 'mutating', 'planning', 'subagent', 'other', 'text', 'dead']

export default function Legend({ hiddenCats, onToggle }) {
  return (
    <div className="legend">
      {ORDER.map(c => (
        <button
          key={c}
          className="chip"
          aria-pressed={!hiddenCats.has(c)}
          onClick={() => onToggle(c)}
        >
          <span className="sw" style={{ background: CATS[c].sw }}></span>
          {CATS[c].label}
        </button>
      ))}
      <span className="hint">click to filter</span>
    </div>
  )
}
