# AVBD Web — WebGPU Physics Engine

## Project Overview

AVBD (Augmented Vertex Block Descent) is a WebGPU-accelerated rigid body physics engine supporting both 2D (3-DOF) and 3D (6-DOF) simulations. It implements the AVBD solver algorithm with both GPU compute shader and CPU reference paths.

## Architecture

```
src/
  2d/index.ts          — Public API: World, RigidBodyDesc2D, ColliderDesc2D
  3d/index.ts          — Public API: World3D, RigidBodyDesc3D, ColliderDesc3D
  core/
    solver.ts          — CPU 2D solver (AVBDSolver2D) with 3×3 LDL^T
    solver-3d.ts       — CPU 3D solver (AVBDSolver3D) with 6×6 LDL^T
    gpu-solver-2d.ts   — GPU 2D solver (WebGPU compute dispatches)
    gpu-solver-3d.ts   — GPU 3D solver (WebGPU compute dispatches)
    types.ts           — Shared types: Vec2/3, Quat, SolverConfig, StepTimings
    rigid-body.ts      — 2D body store, collider descriptors, AABB computation
    rigid-body-3d.ts   — 3D body store
    math.ts            — Linear algebra (mat3, LDL solvers, vector ops)
    graph-coloring.ts  — Graph coloring for parallel GPU dispatch
  constraints/
    constraint.ts      — ConstraintStore with warmstart cache
    contact.ts         — 2D contact constraint creation
    joint.ts, spring.ts, motor.ts — Joint/spring/motor constraints
  2d/collision-sat.ts  — 2D SAT collision (box-box, box-circle, circle-circle)
  3d/collision-gjk.ts  — 3D GJK/EPA collision (box-box, box-sphere, sphere-sphere)
  shaders/             — WGSL compute shaders (primal-update, dual-update, friction-coupling)
examples/
  demo-2d.html         — 2D demo with canvas rendering
  demo-3d.html         — 3D demo with Three.js rendering
```

## Commands

- `npm test` — Run vitest unit tests (497 tests, ~40s)
- `npm run test:gpu` — Run Playwright browser tests (requires WebGPU-capable browser)
- `npm run test:all` — Run both vitest and Playwright tests
- `npm run dev` — Start Vite dev server for demos (http://localhost:5173)
- `npm run build` — TypeScript compilation
- `npm run typecheck` — Type check without emitting
- `npm run lint` — ESLint

## Testing

### Unit Tests (vitest)
- Located in `tests/` directory
- Run with `npx vitest run` or `npx vitest run tests/specific-test.test.ts`
- 2D World uses `world.stepCPU()` (no GPU in vitest)
- 3D World uses `world.step()` with `useCPU: true` option
- Key test files:
  - `tests/repro-pyramid.test.ts` — Pyramid/stack stability acceptance tests
  - `tests/collision-sat.test.ts` — 2D collision detection
  - `tests/collision-edge-cases.test.ts` — Edge case collision scenarios
  - `tests/physics-scenarios.test.ts` — Physics behavior validation
  - `tests/warmstarting.test.ts` — Contact cache warmstarting

### Playwright Tests
- Config: `playwright.config.ts` (Chromium with WebGPU flags)
- Test harness: `tests/browser/test-harness.html`
- Tests: `tests/browser/gpu-execution.spec.ts`
- Dev server auto-starts on port 3333

## Solver Algorithm

Each physics step follows this pipeline:
1. **Broadphase** — Spatial hash grid → candidate pairs (O(n) average)
2. **Narrowphase** — SAT (2D) or GJK (3D) collision → contact manifolds
3. **Warmstart** — Decay penalty/lambda from previous frame
4. **Body Init** — Velocity clamping, adaptive gravity, inertial targets
5. **Solver Loop** (N iterations):
   - Primal update: build & solve LDL^T system per body
   - Dual update: update lambda, ramp penalty, friction coupling
6. **Velocity Recovery** — BDF1 from position difference
7. **Post-stabilization** — Extra primal iteration (no dual)

### GPU Path
Same algorithm but primal/dual updates run as WGSL compute shaders.
Graph coloring ensures conflict-free parallel body updates.
Buffer upload (CPU→GPU) and readback (GPU→CPU) bracket the GPU dispatches.

## Performance Profiling

`world.lastTimings` (or `world3d.lastTimings`) returns a `StepTimings` object:
```typescript
interface StepTimings {
  total, broadphase, narrowphase, warmstart, bodyInit, solverIters, velocityRecover: number;
  bufferUpload?, gpuDispatch?, readback?: number;  // GPU-only
  numBodies, numConstraints: number;
}
```

Both demos display this breakdown in their stats overlay.

## Known Issues & Design Decisions

- **Restitution in 3D**: Property exists on bodies but only takes effect via velocity-level correction in the CPU 3D solver (not in the GPU 3D shader path).
- **Adaptive gravity**: Only applied to slow-moving bodies (speed < 0.5 m/s) to avoid artificial bounce on impact.
- **Implicit angular damping**: 0.05 applied in all solvers to prevent solver-artifact angular velocity amplification in stacking.
