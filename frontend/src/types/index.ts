/** Contrato de datos del frontend. Todo lo que llega del backend / mock se normaliza a estos tipos. */

export interface Subject {
  code: string
  name: string
  department: string
  /** Duración del examen tal como venga en los datos (≤ 8 se muestra como horas, si no minutos). */
  duration?: number
}

/** Mapa turno → materias: { "0": [...], "1": [...] }. El índice del turno es 0-based. */
export type Schedule = Record<string, Subject[]>

export interface Metrics {
  threshold: number
  turnos_usados: number
  max_turnos: number
  /** 0–100 */
  alumnos_protegidos_pct: number
  /** 0–100, precisión en aristas del Árbol de Decisión */
  precision_arbol: number
  precision_knn?: number
  /** Turnos que requeriría el oráculo (inscripciones reales conocidas) */
  turnos_oraculo: number
}

export interface SweepPoint {
  threshold: number
  turnos: number
  alumnos_protegidos_pct: number
}

export interface GraphNode {
  id: string
  label: string
  color?: string
}

export interface GraphEdge {
  from: string
  to: string
  /** Probabilidad de conflicto predicha por el modelo (opcional) */
  prob?: number
  /** Peso de la arista (intensidad de conflicto o probabilidad) */
  weight?: number
  /** Factores curriculares que explican la arista (opcional) */
  factors?: string[]
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}
