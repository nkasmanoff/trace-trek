import { useState, useMemo, useCallback } from 'react'
import { buildTrace, computeMetrics } from '../utils/trace'
import MetaBar from './MetaBar'
import Metrics from './Metrics'
import SystemCard from './SystemCard'
import TurnCard from './TurnCard'
import Legend from './Legend'

function stepKey(ti, si) {
  return `${ti}-${si}`
}

export default function AnatomyView({ store }) {
  const { sessions, activeIdx, setActiveIdx } = store
  const session = sessions[activeIdx]
  const [hiddenCats, setHiddenCats] = useState(new Set())
  const [expandedSteps, setExpandedSteps] = useState({})

  const trace = useMemo(() => session ? buildTrace(session.messages) : null, [session])
  const metrics = useMemo(() => trace && session ? computeMetrics(trace, session.meta) : null, [trace, session])

  const toggleCat = useCallback((cat) => {
    setHiddenCats(prev => {
      const next = new Set(prev)
      if (next.has(cat)) next.delete(cat); else next.add(cat)
      return next
    })
  }, [])

  const toggleStep = useCallback((key, forceOpen) => {
    setExpandedSteps(prev => ({
      ...prev,
      [key]: forceOpen != null ? forceOpen : !prev[key]
    }))
  }, [])

  const expandAll = useCallback(() => {
    if (!trace) return
    const all = {}
    for (let ti = 0; ti < trace.turns.length; ti++) {
      for (let si = 0; si < trace.turns[ti].steps.length; si++) {
        all[stepKey(ti, si)] = true
      }
    }
    setExpandedSteps(all)
  }, [trace])

  const collapseAll = useCallback(() => setExpandedSteps({}), [])

  if (!trace || !metrics) return null

  return (
    <div>
      <MetaBar
        sessions={sessions}
        activeIdx={activeIdx}
        meta={session?.meta}
        onChangeSession={setActiveIdx}
      />
      <Metrics metrics={metrics} />
      <Legend hiddenCats={hiddenCats} onToggle={toggleCat} />
      <div className="controls">
        <button className="btn" onClick={expandAll}>Expand all steps</button>
        <button className="btn" onClick={collapseAll}>Collapse all steps</button>
      </div>
      <SystemCard
        system={trace.system}
        reconstructed={session?.meta?.reconstructedSystem || null}
        tools={session?.meta?.reconstructedTools || null}
      />
      <div id="turns">
        {trace.turns.map((turn, ti) => (
          <TurnCard
            key={ti}
            turn={turn}
            turnIdx={ti}
            hiddenCats={hiddenCats}
            expandedSteps={expandedSteps}
            onToggleStep={toggleStep}
          />
        ))}
      </div>
    </div>
  )
}
