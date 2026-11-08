import { useEffect, useMemo, useRef, useState } from 'react'
import { Network } from 'vis-network/standalone'
import { api } from '../services/apiClient'
import type { GraphEdge, Schedule, Subject } from '../types'
import { fmt, fmtTau, turnBorder, turnColor, turnLabel } from '../utils'

type GroupBy = 'departamento' | 'turno'
type Sub = Subject & { turno: number }

// Tope de nodos para que vis-network se mantenga fluido (el grafo completo tiene ~800 nodos).
const CORE_MAX: Record<GroupBy, number> = { departamento: 50, turno: 20 }
const VIEW_MAX = 50

interface Props {
  schedule: Schedule | null
  tau: number
  focus: string | null
  onFocus: (code: string | null) => void
}

export default function GraphView({ schedule, tau, focus, onFocus }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const net = useRef<Network | null>(null)
  const focusCb = useRef(onFocus)
  focusCb.current = onFocus

  const [groupBy, setGroupBy] = useState<GroupBy>('departamento')
  const [value, setValue] = useState('')
  const [pick, setPick] = useState('')
  const [edges, setEdges] = useState<GraphEdge[] | null>(null)
  const [err, setErr] = useState<string | null>(null)

  // código → materia con su turno, armado desde el cronograma
  const idx = useMemo(() => {
    const m = new Map<string, Sub>()
    Object.entries(schedule ?? {}).forEach(([k, list]) => list.forEach((s) => m.set(s.code, { ...s, turno: Number(k) })))
    return m
  }, [schedule])

  const options = useMemo(
    () =>
      groupBy === 'turno'
        ? Object.keys(schedule ?? {}).sort((a, b) => Number(a) - Number(b))
        : [...new Set([...idx.values()].map((s) => s.department))].sort((a, b) => a.localeCompare(b, 'es')),
    [groupBy, schedule, idx],
  )
  useEffect(() => {
    if (!options.includes(value)) setValue(options[0] ?? '')
  }, [options, value])

  const datalist = useMemo(
    () => [...idx.values()].map((s) => <option key={s.code} value={s.code}>{s.name}</option>),
    [idx],
  )

  // Aristas: grafo completo, o sólo las de la materia inspeccionada
  useEffect(() => {
    let off = false
    setEdges(null)
    setErr(null)
    api.getGraph(tau, focus ?? undefined)
      .then((g) => !off && setEdges(g.edges))
      .catch((e) => !off && setErr(e instanceof Error ? e.message : String(e)))
    return () => { off = true }
  }, [tau, focus])

  // Subgrafo que se dibuja
  const view = useMemo(() => {
    if (!edges) return null
    if (focus) {
      const es = edges.filter((e) => e.from === focus || e.to === focus)
      const ids = new Set([focus, ...es.flatMap((e) => [e.from, e.to])])
      return { ids: [...ids], es, note: '' }
    }
    const deg = new Map<string, number>()
    edges.forEach((e) => {
      deg.set(e.from, (deg.get(e.from) ?? 0) + 1)
      deg.set(e.to, (deg.get(e.to) ?? 0) + 1)
    })
    const core = [...idx.values()]
      .filter((s) => (groupBy === 'turno' ? String(s.turno) === value : s.department === value))
      .map((s) => s.code)
      .sort((a, b) => (deg.get(b) ?? 0) - (deg.get(a) ?? 0))
    const ids = new Set(core.slice(0, CORE_MAX[groupBy]))
    if (groupBy === 'turno') {
      // vecinos de otros turnos: dejan ver que las aristas siempre unen colores distintos
      const freq = new Map<string, number>()
      edges.forEach((e) => {
        const a = ids.has(e.from), b = ids.has(e.to)
        if (a !== b) {
          const o = a ? e.to : e.from
          freq.set(o, (freq.get(o) ?? 0) + 1)
        }
      })
      ;[...freq].sort((x, y) => y[1] - x[1]).slice(0, VIEW_MAX - ids.size).forEach(([c]) => ids.add(c))
    }
    const es = edges.filter((e) => ids.has(e.from) && ids.has(e.to))
    const note = core.length > CORE_MAX[groupBy]
      ? `Se muestran ${CORE_MAX[groupBy]} de ${core.length} materias del grupo, las que tienen más conflictos.`
      : ''
    return { ids: [...ids], es, note }
  }, [edges, focus, groupBy, value, idx])

  // Instancia única de vis-network
  useEffect(() => {
    const n = new Network(box.current!, { nodes: [], edges: [] }, {
      nodes: { shape: 'dot', size: 13, borderWidth: 1.5, font: { size: 11, face: 'Public Sans, sans-serif', color: '#1b2a26' } },
      edges: { color: { color: '#9aa8ad', highlight: '#1b2a26', hover: '#1b2a26' }, width: 1, smooth: false },
      physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -60, springLength: 95 }, stabilization: { iterations: 200 } },
      interaction: { hover: true, tooltipDelay: 120 },
    })
    n.on('click', (p) => { if (p?.nodes?.length) focusCb.current(String(p.nodes[0])) })
    net.current = n
    return () => { n.destroy(); net.current = null }
  }, [])

  const turnoOf = (c: string) => idx.get(c)?.turno
  const mono = (e: GraphEdge) => turnoOf(e.from) !== undefined && turnoOf(e.from) === turnoOf(e.to)

  useEffect(() => {
    if (!view || !net.current) return
    net.current.setData({
      nodes: view.ids.map((c) => {
        const s = idx.get(c)
        const t = s?.turno ?? 0
        const main = c === focus
        return {
          id: c,
          label: c,
          title: s ? `${s.name}\n${s.department}, ${turnLabel(t)}` : c,
          size: main ? 24 : 13,
          borderWidth: main ? 4 : 1.5,
          font: { size: main ? 14 : 11 },
          color: {
            background: turnColor(t), border: turnBorder(t),
            highlight: { background: turnColor(t), border: '#1b2a26' },
            hover: { background: turnColor(t), border: '#1b2a26' },
          },
        }
      }),
      edges: view.es.map((e) => ({
        id: `${e.from}|${e.to}`,
        from: e.from,
        to: e.to,
        width: mono(e) ? 4 : 1 + (e.prob ?? 0.3) * 3,
        title: e.prob !== undefined ? `Probabilidad de choque: ${fmt(e.prob)}` : undefined,
        ...(mono(e) ? { color: { color: '#b3261e' } } : {}),
      })),
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, idx, focus])

  const choose = (v: string) => {
    setPick(v)
    if (idx.has(v)) { onFocus(v); setPick('') }
  }

  const sel = focus ? idx.get(focus) : undefined
  const bad = view ? view.es.filter(mono).length : 0
  const turnos = view ? [...new Set(view.ids.map((c) => turnoOf(c) ?? 0))].sort((a, b) => a - b) : []
  const nb = focus && view
    ? view.es
        .map((e) => ({ code: e.from === focus ? e.to : e.from, e }))
        .sort((a, b) => (b.e.prob ?? 0) - (a.e.prob ?? 0) || a.code.localeCompare(b.code))
    : []

  return (
    <section>
      <div className="bar-tools">
        {focus ? (
          <>
            <span className="lbl">Inspeccionando la materia</span>
            <b>{focus}</b>
            <button className="btn" onClick={() => onFocus(null)}>Volver al subgrafo</button>
          </>
        ) : (
          <>
            <div className="seg" role="group" aria-label="Agrupar subgrafo">
              {(['departamento', 'turno'] as GroupBy[]).map((g) => (
                <button key={g} className={g === groupBy ? 'on' : ''} onClick={() => setGroupBy(g)}>Por {g}</button>
              ))}
            </div>
            <select value={value} onChange={(e) => setValue(e.target.value)} aria-label={`Elegir ${groupBy}`}>
              {options.map((o) => <option key={o} value={o}>{groupBy === 'turno' ? turnLabel(o) : o}</option>)}
            </select>
          </>
        )}
        <input list="subj-list" className="grow" placeholder="Inspeccionar una materia: código o nombre" value={pick}
               onChange={(e) => choose(e.target.value)} aria-label="Inspeccionar materia" />
        <datalist id="subj-list">{datalist}</datalist>
      </div>

      <div className="gv">
        <div className="canvas">
          <div ref={box} className="vis" />
          {!view && !err && <div className="overlay">Cargando grafo…</div>}
          {err && <div className="overlay err">{err}</div>}
        </div>

        <aside className="side">
          {sel && (
            <div className="sel">
              <span className="chip" style={{ background: turnColor(sel.turno) }}>{turnLabel(sel.turno)}</span>
              <h3>{sel.name}</h3>
              <p>{sel.code}, {sel.department}</p>
            </div>
          )}
          <dl className="facts">
            <div><dt>Materias en pantalla</dt><dd>{view ? view.ids.length : '—'}</dd></div>
            <div><dt>Aristas de conflicto</dt><dd>{view ? view.es.length : '—'}</dd></div>
            <div>
              <dt>Aristas con ambos extremos en el mismo turno</dt>
              <dd className={view ? (bad ? 'bad' : 'ok') : ''}>{view ? bad : '—'}</dd>
            </div>
          </dl>
          {view?.note && <p className="hint">{view.note}</p>}

          {focus ? (
            <>
              <p className="expl">
                Estas {nb.length} materias están unidas a <b>{focus}</b> porque el modelo estima una probabilidad de choque de al
                menos {fmtTau(tau)}. Por eso el coloreo greedy no puede ubicarlas en su mismo turno.
              </p>
              <ul className="nb">
                {nb.map(({ code, e }) => {
                  const s = idx.get(code)
                  return (
                    <li key={code}>
                      <button onClick={() => onFocus(code)}>
                        <i style={{ background: turnColor(s?.turno ?? 0) }} />
                        <span><b>{code}</b> {s?.name}</span>
                        <small>{s ? turnLabel(s.turno) : ''}{e.prob !== undefined ? `, p = ${fmt(e.prob)}` : ''}</small>
                        {e.factors && e.factors.length > 0 && <small className="why">{e.factors.join(', ')}</small>}
                      </button>
                    </li>
                  )
                })}
              </ul>
            </>
          ) : (
            <p className="expl">Hacé clic en una materia del lienzo para aislarla y ver solo sus conflictos directos.</p>
          )}

          <div className="legend">
            {turnos.map((t) => <span key={t}><i style={{ background: turnColor(t) }} />{turnLabel(t)}</span>)}
          </div>
        </aside>
      </div>
    </section>
  )
}
