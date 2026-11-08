import { useEffect, useState } from 'react'
import GraphView from './components/GraphView'
import MetricCards from './components/MetricCards'
import ScheduleView from './components/ScheduleView'
import ThresholdSlider from './components/ThresholdSlider'
import { api, DEFAULT_TAU } from './services/apiClient'
import type { Metrics, Schedule, SweepPoint } from './types'

type Tab = 'cronograma' | 'grafo'

export default function App() {
  const [tau, setTau] = useState(DEFAULT_TAU) // valor del slider (inmediato)
  const [qTau, setQTau] = useState(DEFAULT_TAU) // valor con el que se consulta (con debounce)
  const [schedule, setSchedule] = useState<Schedule | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [sweep, setSweep] = useState<SweepPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)
  const [tab, setTab] = useState<Tab>('cronograma')
  const [inspect, setInspect] = useState<string | null>(null)

  useEffect(() => {
    const t = setTimeout(() => setQTau(tau), 250)
    return () => clearTimeout(t)
  }, [tau])

  useEffect(() => {
    let off = false
    setLoading(true)
    Promise.all([api.getSchedule(qTau), api.getStats(qTau)])
      .then(([s, m]) => {
        if (off) return
        setSchedule(s)
        setMetrics(m)
        setError(null)
      })
      .catch((e) => !off && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !off && setLoading(false))
    return () => { off = true }
  }, [qTau, retry])

  useEffect(() => {
    api.getSweep().then(setSweep).catch(() => setSweep([]))
  }, [retry])

  const live = api.mode === 'api'

  return (
    <>
      <div className="board">
        <div className="wrap">
          <div className="top">
            <div>
              <h1>Planificación de mesas de examen</h1>
              <p className="lede">
                Cada materia es un nodo del grafo. Dos materias unidas por una arista no pueden rendirse en el mismo
                turno, así que el coloreo les asigna turnos distintos.
              </p>
            </div>
            <span className={`pill ${live ? 'live' : ''}`}>{live ? 'Conectado al backend' : 'Modo offline, datos precalculados'}</span>
          </div>
          <ThresholdSlider tau={tau} onChange={setTau} metrics={metrics} sweep={sweep}
                           pending={tau !== qTau || loading} stale={!api.exact(tau)} />
        </div>
      </div>

      <main className="wrap">
        {error && (
          <div className="alert" role="alert">
            <p>
              <b>No se pudieron cargar los datos.</b> {error}
              {live && ' Verificá que el backend esté levantado y que VITE_API_URL apunte a él.'}
            </p>
            <button className="btn" onClick={() => setRetry((n) => n + 1)}>Reintentar</button>
          </div>
        )}

        <MetricCards m={metrics} />

        <nav className="tabs" aria-label="Vistas">
          <button className={tab === 'cronograma' ? 'on' : ''} onClick={() => setTab('cronograma')}>Cronograma de turnos</button>
          <button className={tab === 'grafo' ? 'on' : ''} onClick={() => setTab('grafo')}>Grafo de conflictos</button>
        </nav>

        {tab === 'cronograma' ? (
          <ScheduleView schedule={schedule} onInspect={(c) => { setInspect(c); setTab('grafo') }} />
        ) : (
          <GraphView schedule={schedule} tau={qTau} focus={inspect} onFocus={setInspect} />
        )}
      </main>

      <footer className="wrap foot">
        Trabajo Práctico Integrador de Lógica y Teoría de Grafos + Machine Learning. Grupo Antonio – Mamani.
      </footer>
    </>
  )
}
