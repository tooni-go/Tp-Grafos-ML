function hsl2hex(h: number, s: number, l: number) {
  s /= 100
  l /= 100
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) =>
    Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))))
  return '#' + [f(0), f(8), f(4)].map((v) => v.toString(16).padStart(2, '0')).join('')
}

// Ángulo áureo: turnos consecutivos quedan con tonos bien distintos, sirve para 11 o para 40 turnos.
const hue = (t: number) => (t * 137.508 + 18) % 360
export const turnColor = (t: number) => hsl2hex(hue(t), 52, 62)
export const turnBorder = (t: number) => hsl2hex(hue(t), 55, 34)
export const turnLabel = (k: string | number) => `Turno ${Number(k) + 1}`

const nf = (d: number) =>
  new Intl.NumberFormat('es-AR', { minimumFractionDigits: d, maximumFractionDigits: d })
export const fmt = (n: number, d = 2) => nf(d).format(n)
export const fmtTau = (t: number) => fmt(t, 2)

/** Normaliza texto para buscar sin importar tildes ni mayúsculas. */
export const plain = (s: string) => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
