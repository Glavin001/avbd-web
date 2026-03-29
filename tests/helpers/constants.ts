/**
 * Shared constants for all test tiers (vitest unit + Playwright browser).
 */

// ─── Scene lists (match the <select> dropdowns in demo pages) ───────────

export const SCENES_2D = [
  'stack', 'pyramid', 'rope', 'heavyRope', 'friction', 'fracture',
  'jointGrid', 'boxRain200', 'largePyramid', 'ballPit300', 'ballPit1000',
  'ballPit2000', 'ballPit10000',
] as const;

export const SCENES_3D = [
  'ground', 'stack', 'pyramid', 'friction', 'boxRain200',
  'largePyramid', 'spherePile100', 'ballPit1000', 'ballPit2000',
  'ballPit10000',
] as const;

// Scenes small enough for longer simulation runs
export const FAST_SCENES_2D: readonly string[] = [
  'stack', 'pyramid', 'rope', 'heavyRope', 'friction',
  'fracture', 'jointGrid', 'boxRain200', 'largePyramid', 'ballPit300',
];
export const FAST_SCENES_3D: readonly string[] = [
  'ground', 'stack', 'pyramid', 'friction', 'boxRain200',
  'largePyramid', 'spherePile100', 'ballPit1000',
];

// Large scenes — shorter sim time to keep test suite fast
export const LARGE_SCENES_2D: readonly string[] = [
  'ballPit1000', 'ballPit2000', 'ballPit10000',
];
export const LARGE_SCENES_3D: readonly string[] = [
  'ballPit2000', 'ballPit10000',
];

// ─── Physics bounds ─────────────────────────────────────────────────────

export const WORLD_BOUNDS_2D = { minX: -50, maxX: 50, minY: -10, maxY: 500 };
export const WORLD_BOUNDS_3D = { minX: -100, maxX: 100, minY: -10, maxY: 500, minZ: -100, maxZ: 100 };

// Ground cuboid center at y=0, half-height=0.5, so surface is at y=0.5
export const GROUND_SURFACE_Y = 0.5;

// Bodies faster than this are considered "exploded"
export const MAX_REASONABLE_SPEED = 500; // m/s

// ─── URLs ───────────────────────────────────────────────────────────────

export const BASE_URL = 'http://localhost:3333/examples';
export const DEMO_2D_URL = `${BASE_URL}/demo-2d.html`;
export const DEMO_3D_URL = `${BASE_URL}/demo-3d.html`;
