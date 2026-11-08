/**
 * Cliente desacoplado (Contract-First).
 *  - Sin VITE_API_URL  → modo "mock": lee los JSON de src/mock (precalculados para τ = 0.20).
 *  - Con VITE_API_URL  → modo "api": GET /api/v1/schedule | /api/v1/schedule/evaluate | /api/v1/graph | /api/v1/schedule/sweep
 *
 * Los componentes sólo ven los tipos de ../types. Si los JSON reales tienen otras claves,
 * se adaptan SOLAMENTE las funciones toSubject / toMetrics / toSweep / toGraph de este archivo.
 */
import type { GraphData, GraphEdge, Metrics, Schedule, Subject, SweepPoint } from '../types'

type R = Record<string, unknown>

const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')
const API_MODE: 'api' | 'mock' = API ? 'api' : 'mock'
export const DEFAULT_TAU = 0.2
const MOCK_TAU = 0.2

// Los archivos de src/mock son opcionales (graph y sweep pueden faltar sin romper el build).
const files = import.meta.glob('../mock/*.json', { eager: true, import: 'default' }) as Record<string, unknown>
const mockFile = (name: string) => files[`../mock/${name}.json`]

/* ───────── normalizadores: ADAPTAR ACÁ si cambian las claves del JSON real ───────── */

const pick = (o: R, ...keys: string[]) => {
  for (const k of keys) if (o[k] !== undefined && o[k] !== null) return o[k]
  return undefined
}
const num = (v: unknown) => {
  const n = Number(v)
  return v === undefined || v === null || v === '' || Number.isNaN(n) ? undefined : n
}
/** Acepta 0.8702 o 87.02 y devuelve siempre 0–100. */
const pct = (v: unknown) => {
  const n = num(v)
  return n === undefined ? 0 : n <= 1 ? n * 100 : n
}

function toSubject(x: unknown): Subject {
  if (typeof x === 'string') return { code: x, name: x, department: '—' }
  const o = x as R
  const code = String(pick(o, 'code', 'codigo', 'cod', 'id', 'exam_id') ?? '?')
  return {
    code,
    name: String(pick(o, 'name', 'nombre', 'asignatura', 'label', 'exam_name') ?? code),
    department: String(pick(o, 'department', 'departamento', 'depto', 'dept', 'department_code') ?? '—'),
    duration: num(pick(o, 'duration_minutes', 'duration', 'duracion', 'duracion_min', 'duracion_horas')),
  }
}

function toSchedule(raw: unknown): Schedule {
  const o = (raw ?? {}) as R
  const m = (o.schedule ?? o.turnos ?? o) as R
  const out: Schedule = {}
  for (const [k, v] of Object.entries(m)) if (Array.isArray(v)) out[k] = v.map(toSubject)
  return out
}

function toMetrics(raw: unknown, tau: number): Metrics {
  const r = (raw ?? {}) as R
  const m = (r.metrics ?? r) as R
  const ev = (r.evaluacion_vs_real ?? m.evaluacion_vs_real ?? m) as R
  const oracle = (r.referencia_oraculo ?? m.referencia_oraculo ?? m.oraculo ?? r.oraculo) as R | undefined
  const knn = pick(m, 'precision_knn', 'knn_precision') ?? 22.41

  const choquesPct = num(pick(ev, 'pct_alumnos_en_choque', 'choques_pct', 'clash_student_percentage'))
  const rawProtegidos = pick(m, 'alumnos_protegidos_pct', 'alumnos_protegidos', 'pct_protegidos') ??
    (choquesPct !== undefined ? 100 - choquesPct : undefined)

  return {
    threshold: num(pick(m, 'threshold', 'umbral', 'tau')) ?? num(r.threshold) ?? tau,
    turnos_usados: num(pick(ev, 'turnos', 'turnos_usados', 'n_turnos', 'turns_count')) ?? num(pick(m, 'turnos_usados', 'turnos', 'turns_count')) ?? 0,
    max_turnos: num(pick(m, 'max_turnos', 'tope_turnos', 'max_allowed_turns')) ?? 32,
    alumnos_protegidos_pct: pct(rawProtegidos),
    precision_arbol: pct(pick(ev, 'precision_aristas', 'precision', 'edge_precision') ?? pick(m, 'precision_arbol', 'precision', 'edge_precision')),
    precision_knn: knn === undefined ? undefined : pct(knn),
    turnos_oraculo:
      num(typeof oracle === 'object' && oracle ? pick(oracle as R, 'turnos', 'turnos_usados') : oracle) ?? 19,
  }
}

function toSweep(raw: unknown): SweepPoint[] {
  const arr = Array.isArray(raw) ? raw : (((raw ?? {}) as R).sweep as unknown[] | undefined) ?? []
  return arr
    .map((x) => {
      const o = x as R
      return {
        threshold: num(pick(o, 'threshold', 'tau', 'umbral')),
        turnos: num(pick(o, 'turnos', 'turnos_usados', 'n_turnos')),
        alumnos_protegidos_pct: pct(pick(o, 'alumnos_protegidos_pct', 'alumnos_protegidos', 'pct_protegidos')),
      }
    })
    .filter((p): p is SweepPoint => p.threshold !== undefined && p.turnos !== undefined)
    .sort((a, b) => a.threshold - b.threshold)
}

function toGraph(raw: unknown): GraphData {
  const o = (raw ?? {}) as R
  const edges: GraphEdge[] = ((o.edges as unknown[]) ?? []).map((x) => {
    const e = x as R
    const f = pick(e, 'factors', 'factores')
    return {
      from: String(pick(e, 'from', 'source', 'u')),
      to: String(pick(e, 'to', 'target', 'v')),
      prob: num(pick(e, 'prob', 'probability', 'weight', 'p')),
      weight: num(pick(e, 'weight', 'prob', 'probability', 'p')),
      factors: Array.isArray(f) ? f.map(String) : undefined,
    }
  })
  const rawNodes = ((o.nodes as unknown[]) ?? []) as unknown[]
  let nodes = rawNodes.map((x) => {
    const n = x as R
    const id = String(pick(n, 'id', 'code', 'exam_id') ?? '?')
    return { id, label: String(pick(n, 'label', 'name', 'exam_name') ?? id), color: pick(n, 'color') as string | undefined }
  })
  if (nodes.length === 0 && edges.length > 0) {
    const nodeIds = new Set<string>()
    for (const e of edges) {
      if (e.from) nodeIds.add(e.from)
      if (e.to) nodeIds.add(e.to)
    }
    nodes = Array.from(nodeIds).map((id) => ({ id, label: id, color: undefined }))
  }
  return { nodes, edges }
}

/* ───────── transporte ───────── */

async function http(path: string, params: Record<string, string | number | boolean | undefined> = {}) {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined) qs.set(k, String(v))
  const queryStr = qs.toString() ? `?${qs.toString()}` : ''
  const res = await fetch(`${API}${path}${queryStr}`)
  if (!res.ok) throw new Error(`${path} respondió HTTP ${res.status}`)
  return res.json() as Promise<unknown>
}

const neighborhood = (g: GraphData, s: string): GraphData => {
  const edges = g.edges.filter((e) => e.from === s || e.to === s)
  const ids = new Set([s, ...edges.flatMap((e) => [e.from, e.to])])
  return { nodes: g.nodes.filter((n) => ids.has(n.id)), edges }
}
const graphCache = new Map<number, GraphData>()

export const api = {
  mode: API_MODE,

  /** ¿Los datos mostrados corresponden realmente a este τ? (en mock sólo existe τ = 0.20) */
  exact: (tau: number) => API_MODE === 'api' || Math.abs(tau - MOCK_TAU) < 1e-9,

  async getSchedule(tau: number): Promise<Schedule> {
    return toSchedule(API ? await http('/api/v1/schedule', { threshold: tau }) : mockFile('schedule_0_20'))
  },

  async getStats(tau: number): Promise<Metrics> {
    return toMetrics(API ? await http('/api/v1/schedule/evaluate', { threshold: tau }) : mockFile('summary'), tau)
  },

  async getSweep(): Promise<SweepPoint[]> {
    return toSweep(API ? await http('/api/v1/schedule/sweep') : (mockFile('sweep') ?? mockFile('summary')))
  },

  /** Sin `subject`: grafo completo. Con `subject`: sólo sus conflictos inmediatos. */
  async getGraph(tau: number, subject?: string): Promise<GraphData> {
    let g = subject ? undefined : graphCache.get(tau)
    if (!g) {
      if (API) {
        g = toGraph(await http('/api/v1/graph', { threshold: tau, include_edges: true }))
      } else {
        const raw = mockFile('graph_0_20')
        if (!raw) throw new Error('Falta src/mock/graph_0_20.json para ver el grafo en modo offline.')
        g = toGraph(raw)
      }
      if (!subject) graphCache.set(tau, g)
    }
    return subject ? neighborhood(g, subject) : g
  },
}
