import { useMemo, useState } from 'react'
import type { Schedule } from '../types'
import { plain, turnBorder, turnColor, turnLabel } from '../utils'

const duration = (d?: number) => (d === undefined ? null : d <= 8 ? `${d} h` : `${d} min`)

interface Props {
  schedule: Schedule | null
  onInspect: (code: string) => void
}

export default function ScheduleView({ schedule, onInspect }: Props) {
  const [q, setQ] = useState('')
  const [dept, setDept] = useState('')

  const keys = useMemo(() => Object.keys(schedule ?? {}).sort((a, b) => Number(a) - Number(b)), [schedule])
  const depts = useMemo(
    () => [...new Set(keys.flatMap((k) => schedule![k].map((s) => s.department)))].sort((a, b) => a.localeCompare(b, 'es')),
    [keys, schedule],
  )

  if (!schedule) return <p className="empty">Cargando cronograma…</p>
  if (!keys.length) return <p className="empty">No hay turnos para mostrar. Revisá que el JSON del cronograma tenga la forma {'{ "0": [materias], "1": [...] }'}.</p>

  const nq = plain(q.trim())
  const filtering = Boolean(nq || dept)
  const cols = keys
    .map((k) => ({
      k,
      items: schedule[k].filter(
        (s) => (!dept || s.department === dept) && (!nq || plain(`${s.code} ${s.name}`).includes(nq)),
      ),
    }))
    .filter((c) => !filtering || c.items.length)
  const total = keys.reduce((n, k) => n + schedule[k].length, 0)
  const shown = cols.reduce((n, c) => n + c.items.length, 0)

  return (
    <section>
      <div className="bar-tools">
        <input type="search" className="grow" placeholder="Buscar por nombre o código de materia" value={q}
               onChange={(e) => setQ(e.target.value)} aria-label="Buscar materia" />
        <select value={dept} onChange={(e) => setDept(e.target.value)} aria-label="Filtrar por departamento">
          <option value="">Todos los departamentos</option>
          {depts.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <span className="count">{shown} de {total} materias en {cols.length} turnos</span>
      </div>

      {!cols.length ? (
        <p className="empty">Ninguna materia coincide con la búsqueda. Probá con otro nombre o quitá el filtro de departamento.</p>
      ) : (
        <div className="cols">
          {cols.map(({ k, items }) => (
            <div className="col" key={k}>
              <header style={{ background: turnColor(Number(k)) }}>
                <h3>{turnLabel(k)}</h3>
                <span>{items.length} {items.length === 1 ? 'materia' : 'materias'}</span>
              </header>
              <div className="col-body">
                {items.map((s) => (
                  <article className="ficha" key={s.code} style={{ borderLeftColor: turnBorder(Number(k)) }}>
                    <span className="code">{s.code}</span>
                    <h4>{s.name}</h4>
                    <div className="meta">
                      <span>{s.department}</span>
                      {duration(s.duration) && <span>{duration(s.duration)}</span>}
                    </div>
                    <button className="link" onClick={() => onInspect(s.code)}>Ver conflictos</button>
                  </article>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
