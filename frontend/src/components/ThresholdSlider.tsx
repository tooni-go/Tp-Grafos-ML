import type { Metrics, SweepPoint } from '../types'
import { DEFAULT_TAU } from '../services/apiClient'
import { fmt, fmtTau, turnColor } from '../utils'

const MIN = 0.05
const MAX = 0.5
const pos = (t: number) => ((t - MIN) / (MAX - MIN)) * 100
const TICKS = Array.from({ length: 10 }, (_, i) => Math.round((0.05 + i * 0.05) * 100) / 100)

interface P { x: number; y: number }

function at(pts: P[], t: number) {
  if (!pts.length) return 0
  if (t <= pts[0].x) return pts[0].y
  for (let i = 1; i < pts.length; i++) {
    if (t <= pts[i].x) {
      const a = pts[i - 1], b = pts[i]
      return a.y + ((t - a.x) / (b.x - a.x || 1)) * (b.y - a.y)
    }
  }
  return pts[pts.length - 1].y
}

function Spark({ title, pts, tau, limit, fmtV }: {
  title: string; pts: P[]; tau: number; limit?: number; fmtV: (v: number) => string
}) {
  if (pts.length < 2) {
    return <figure className="spark"><figcaption><span>{title}</span></figcaption><small>Sin curva de barrido</small></figure>
  }
  const W = 190, H = 74, M = 8
  const ys = pts.map((p) => p.y).concat(limit !== undefined ? [limit] : [])
  const lo = Math.min(...ys), hi = Math.max(...ys)
  const sx = (x: number) => M + ((x - MIN) / (MAX - MIN)) * (W - 2 * M)
  const sy = (y: number) => H - M - ((y - lo) / (hi - lo || 1)) * (H - 2 * M)
  const cur = at(pts, tau)
  return (
    <figure className="spark">
      <figcaption><span>{title}</span><b>{fmtV(cur)}</b></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} según τ`}>
        {limit !== undefined && (
          <>
            <line className="lim" x1={M} x2={W - M} y1={sy(limit)} y2={sy(limit)} />
            <text x={W - M} y={sy(limit) - 3} textAnchor="end">tope {limit}</text>
          </>
        )}
        <polyline points={pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} />
        <line className="cur" x1={sx(tau)} x2={sx(tau)} y1={M} y2={H - M} />
        <circle cx={sx(tau)} cy={sy(cur)} r="4" />
      </svg>
    </figure>
  )
}

/** Los 32 turnos del calendario oficial como casillas: las usadas toman el color de su turno. */
function CalendarStrip({ used, max, oracle }: { used: number; max: number; oracle: number }) {
  const over = used - max
  return (
    <div className="cal">
      <div className="cells" role="img" aria-label={`${used} de ${max} turnos usados`}
           style={{ gridTemplateColumns: `repeat(${max}, 1fr)` }}>
        {Array.from({ length: max }, (_, i) => (
          <i key={i} style={i < used ? { background: turnColor(i) } : undefined} />
        ))}
      </div>
      <div className="cal-foot">
        {oracle > 0 && (
          <span className="orc" style={{ left: `${((Math.min(oracle, max) - 0.5) / max) * 100}%` }}>
            oráculo: {oracle}
          </span>
        )}
        <span className={over > 0 ? 'over' : ''}>
          {over > 0 ? `${over} por encima del tope` : `${max - used} turnos libres`}
        </span>
      </div>
    </div>
  )
}

interface Props {
  tau: number
  onChange: (t: number) => void
  metrics: Metrics | null
  sweep: SweepPoint[]
  /** Hay una consulta en curso para el τ elegido */
  pending: boolean
  /** Modo offline y τ distinto del precalculado */
  stale: boolean
}

export default function ThresholdSlider({ tau, onChange, metrics, sweep, pending, stale }: Props) {
  const fits = metrics ? metrics.turnos_usados <= metrics.max_turnos : null
  const limit = metrics?.max_turnos
  return (
    <section className="tau" aria-label="Umbral de conflicto">
      <div className="tau-read">
        <span className="lbl">Umbral de conflicto</span>
        <output className="big">τ = {fmtTau(tau)}</output>
        <p>Se dibuja una arista cuando el modelo estima una probabilidad de choque mayor o igual a τ.</p>
        <button className="chalk-link" onClick={() => onChange(DEFAULT_TAU)} disabled={tau === DEFAULT_TAU}>
          Volver al óptimo ({fmtTau(DEFAULT_TAU)})
        </button>
      </div>

      <div className="tau-ctl">
        <div className="track">
          <span className="rec" style={{ left: `${pos(DEFAULT_TAU)}%` }}>óptimo en validación</span>
          <input type="range" min={MIN} max={MAX} step={0.01} value={tau} aria-label="Umbral tau"
                 onChange={(e) => onChange(Math.round(+e.target.value * 100) / 100)} />
          <div className="ticks">
            {TICKS.map((t) => <span key={t} style={{ left: `${pos(t)}%` }}>{fmtTau(t)}</span>)}
          </div>
        </div>

        {metrics && <CalendarStrip used={metrics.turnos_usados} max={metrics.max_turnos} oracle={metrics.turnos_oraculo} />}

        <div className="status">
          {fits !== null && metrics && (
            <span className={`badge ${fits ? 'ok' : 'bad'}`}>
              {fits ? `Cabe en el calendario oficial de ${limit} turnos` : `No cabe: ${metrics.turnos_usados} turnos, el tope es ${limit}`}
            </span>
          )}
          {pending && <span className="hint">Recalculando…</span>}
          {stale && (
            <span className="hint warn">
              Modo offline: solo hay datos precalculados para τ = 0,20. Conectá el backend para recalcular.
            </span>
          )}
        </div>
      </div>

      <div className="tau-sparks">
        <Spark title="Turnos requeridos" pts={sweep.map((p) => ({ x: p.threshold, y: p.turnos }))}
               tau={tau} limit={limit} fmtV={(v) => fmt(v, 0)} />
        <Spark title="Alumnos con choque" pts={sweep.map((p) => ({ x: p.threshold, y: 100 - p.alumnos_protegidos_pct }))}
               tau={tau} fmtV={(v) => `${fmt(v, 1)} %`} />
      </div>
    </section>
  )
}
