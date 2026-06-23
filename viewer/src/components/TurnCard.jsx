import { useState } from 'react'
import StepDetail from './StepDetail'

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

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;')
}

function truncate(s, n) {
  s = String(s || '')
  return s.length > n ? s.slice(0, n) : s
}

function stepCatKey(s) {
  if (s.type === 'thinking') return 'thinking'
  if (s.type === 'text') return 'text'
  return s.cat
}

export default function TurnCard({ turn, turnIdx, hiddenCats, expandedSteps, onToggleStep }) {
  const [closed, setClosed] = useState(false)
  const userPreview = truncate(turn.userText, 280)

  return (
    <article className={`turn${closed ? ' closed' : ''}`}>
      <button className="turn-head" onClick={() => setClosed(!closed)} aria-expanded={!closed}>
        <span className="turn-tag">TURN {turnIdx + 1}</span>
        <span className="turn-msg">
          {esc(userPreview)}
          {turn.userText.length > 280 && <span className="more">…</span>}
        </span>
        <span className="turn-count">{turn.steps.length} steps</span>
        <span className="caret">▾</span>
      </button>
      <ol className="steps">
        {turn.steps.map((step, si) => {
          const k = `${turnIdx}-${si}`
          const catKey = stepCatKey(step)
          const cat = CATS[catKey] || CATS.other
          const dead = step.type === 'tool' && step.dead
          const hide = hiddenCats.has(catKey) || (dead && hiddenCats.has('dead'))

          let label, summary
          if (step.type === 'thinking') { label = 'thinking'; summary = step.text }
          else if (step.type === 'text') { label = 'text'; summary = step.text }
          else { label = step.name; summary = step.summary || '' }

          const tags = []
          if (step.type === 'tool' && step.parallel) tags.push(<span className="badge" key="p">∥ parallel</span>)
          if (dead) tags.push(<span className="badge dead" key="d">dead end</span>)

          const expanded = expandedSteps[k] || false

          return (
            <li
              key={si}
              className={`step${expanded ? ' open' : ''}${hide ? ' hidden-cat' : ''}`}
              data-cat={catKey}
              data-dead={dead ? '1' : ''}
            >
              <button
                className="step-head"
                aria-expanded={expanded}
                onClick={() => onToggleStep(k)}
              >
                <span className="dot" style={{ background: dead ? 'var(--c-dead)' : cat.sw }}></span>
                <span className="step-body">
                  <span className="step-label" style={{ color: dead ? 'var(--c-dead-ink)' : cat.ink }}>
                    {esc(label)}
                  </span>
                  {tags}
                  <span className="step-sum">{esc(truncate(summary, 220))}</span>
                </span>
                <span className="caret">▾</span>
              </button>
              {expanded && (
                <div className="step-detail">
                  <StepDetail step={step} />
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </article>
  )
}
