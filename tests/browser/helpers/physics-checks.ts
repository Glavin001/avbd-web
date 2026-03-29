/**
 * In-browser physics sanity checks.
 *
 * These functions are designed to be passed to `page.evaluate()` — they run
 * inside the browser context, read from the `window.__avbd` test hooks, and
 * return plain serializable objects.
 */

export interface PhysicsSanityResult {
  ok: boolean;
  error?: string;
  bodyCount: number;
  nanCount: number;
  escapedCount: number;
  explodedCount: number;
  penetratingCount: number;
  firstNaN: { x: number; y: number } | null;
  firstEscaped: { x: number; y: number } | null;
}

export interface SanityCheckOpts {
  bounds: { minX: number; maxX: number; minY: number; maxY: number; minZ?: number; maxZ?: number };
  maxSpeed: number;
  groundY: number;
}

/**
 * Inline function to evaluate inside the browser.
 * Checks positions & velocities from __avbd hooks for NaN, escapes, explosions, penetration.
 *
 * Usage:
 *   const result = await page.evaluate(runPhysicsSanityCheck, opts);
 */
export function runPhysicsSanityCheck(opts: SanityCheckOpts): PhysicsSanityResult {
  const avbd = (window as any).__avbd;
  if (!avbd) {
    return {
      ok: false, error: 'No __avbd test hooks found on window',
      bodyCount: 0, nanCount: 0, escapedCount: 0,
      explodedCount: 0, penetratingCount: 0,
      firstNaN: null, firstEscaped: null,
    };
  }

  const positions: { x: number; y: number; z?: number }[] = avbd.getBodyPositions();
  const velocities: { x: number; y: number; z?: number }[] = avbd.getBodyVelocities();

  // NaN / Infinity check
  const nanPositions = positions.filter(
    (p) => !isFinite(p.x) || !isFinite(p.y) || (p.z !== undefined && !isFinite(p.z)),
  );

  // Escape check — body outside world bounds
  const escaped = positions.filter((p) => {
    if (p.x < opts.bounds.minX || p.x > opts.bounds.maxX) return true;
    if (p.y < opts.bounds.minY || p.y > opts.bounds.maxY) return true;
    if (p.z !== undefined && opts.bounds.minZ !== undefined) {
      if (p.z < opts.bounds.minZ || p.z > (opts.bounds.maxZ ?? Infinity)) return true;
    }
    return false;
  });

  // Explosion check — unreasonable velocity
  const exploded = velocities.filter((v) => {
    const speed = Math.sqrt(v.x * v.x + v.y * v.y + (v.z ?? 0) * (v.z ?? 0));
    return speed > opts.maxSpeed;
  });

  // Ground penetration — body significantly below ground surface
  const penetrating = positions.filter((p) => p.y < opts.groundY - 1.0);

  return {
    ok:
      nanPositions.length === 0 &&
      escaped.length === 0 &&
      exploded.length === 0 &&
      penetrating.length === 0,
    bodyCount: positions.length,
    nanCount: nanPositions.length,
    escapedCount: escaped.length,
    explodedCount: exploded.length,
    penetratingCount: penetrating.length,
    firstNaN: nanPositions[0] ?? null,
    firstEscaped: escaped[0] ?? null,
  };
}
