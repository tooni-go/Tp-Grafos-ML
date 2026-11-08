import type { Metrics } from '../types'
import { fmt } from '../utils'

function Bar({ value, tone }: { value: number; tone?: 'alt' }) {
  return <i className={`bar ${tone ?? ''}`}><u style={{ width: `${Math.max(0, Math.min(100, value))}%` }} /></i>
}

export default function MetricCards({ m }: { m: Metrics | null }) {
  if (!m) return <section className="ledger loading" aria-busy="true">Cargando métricas…</section>
  return (
    <section className="ledger" aria-label="Indicadores de la planificación">
      <div className="cell">
        <span className="lbl">Turnos requeridos</span>
        <b className="num">{m.turnos_usados}</b>
        <p>Tope reglamentario: {m.max_turnos} turnos.</p>
      </div>
      <div className="cell">
        <span className="lbl">Alumnos protegidos</span>
        <b className="num">{fmt(m.alumnos_protegidos_pct)} %</b>
        <Bar value={m.alumnos_protegidos_pct} />
        <p>De la cohorte no tiene ninguna colisión horaria.</p>
      </div>
      <div className="cell">
        <span className="lbl">Precisión en aristas</span>
        <b className="num">{fmt(m.precision_arbol)} %</b>
        <Bar value={m.precision_arbol} />
        <p>
          Árbol de Decisión
          {m.precision_knn !== undefined && <> contra {fmt(m.precision_knn)} % de KNN</>}.
        </p>
        {m.precision_knn !== undefined && <Bar value={m.precision_knn} tone="alt" />}
      </div>
      <div className="cell">
        <span className="lbl">Referencia del oráculo</span>
        <b className="num">{m.turnos_oraculo}</b>
        <p>Turnos necesarios si se conocieran las inscripciones reales de antemano.</p>
      </div>
    </section>
  )
}
