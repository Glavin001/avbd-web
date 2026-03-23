# AVBD-Web

**WebGPU-accelerated Augmented Vertex Block Descent (AVBD) physics engine for the web.**

The first GPU-accelerated web implementation of the AVBD algorithm (SIGGRAPH 2025), supporting both 2D and 3D rigid body simulation with a Rapier-compatible API.

## Features

- **AVBD Solver**: Full implementation of the Augmented Vertex Block Descent algorithm with augmented Lagrangian formulation
- **2D + 3D**: Unified codebase supporting both 2D (3-DOF) and 3D (6-DOF) rigid body simulation
- **WebGPU Compute Shaders**: WGSL shaders for GPU-parallel primal/dual updates with graph coloring dispatch
- **Rapier-compatible API**: Familiar `World`, `RigidBodyDesc`, `ColliderDesc`, `JointData` pattern
- **Collision Detection**: SAT (2D boxes), GJK/SAT (3D boxes), sphere-sphere, box-sphere
- **Constraints**: Contact manifolds with friction, revolute joints, fixed joints, distance springs
- **Graph Coloring**: CPU-side greedy coloring for conflict-free parallel GPU dispatch

## Quick Start

```typescript
// 2D Physics
import AVBD from 'avbd-web/2d';

await AVBD.init();
const world = new AVBD.World({ x: 0, y: -9.81 });

// Ground
world.createCollider(AVBD.ColliderDesc.cuboid(10, 0.5));

// Dynamic box
const body = world.createRigidBody(
  AVBD.RigidBodyDesc.dynamic().setTranslation(0, 5)
);
world.createCollider(AVBD.ColliderDesc.cuboid(0.5, 0.5), body);

// Simulate
world.step();
console.log(body.translation()); // { x: 0, y: ~4.997 }
```

```typescript
// 3D Physics
import AVBD3D from 'avbd-web/3d';

await AVBD3D.init();
const world = new AVBD3D.World({ x: 0, y: -9.81, z: 0 });

world.createCollider(AVBD3D.ColliderDesc.cuboid(10, 0.5, 10));

const body = world.createRigidBody(
  AVBD3D.RigidBodyDesc.dynamic().setTranslation(0, 5, 0)
);
world.createCollider(AVBD3D.ColliderDesc.cuboid(0.5, 0.5, 0.5), body);

world.step();
console.log(body.translation()); // { x: 0, y: ~4.997, z: 0 }
```

## Architecture

```
src/
├── core/
│   ├── solver.ts          # 2D AVBD solver (CPU reference)
│   ├── solver-3d.ts       # 3D AVBD solver (CPU reference)
│   ├── gpu-context.ts     # WebGPU device/buffer management
│   ├── graph-coloring.ts  # Greedy coloring for parallel dispatch
│   ├── rigid-body.ts      # 2D body store (struct-of-arrays)
│   ├── rigid-body-3d.ts   # 3D body store
│   ├── math.ts            # Vec2/Vec3/Quat/Mat3 operations, LDL solver
│   └── types.ts           # Shared types and config interfaces
├── shaders/
│   ├── primal-update-2d.wgsl  # 3×3 LDL primal solve
│   ├── primal-update-3d.wgsl  # 6×6 LDL primal solve
│   ├── dual-update.wgsl       # Lambda/penalty update
│   └── math-utils.wgsl        # WGSL matrix operations
├── constraints/
│   ├── constraint.ts      # Constraint row store
│   ├── contact.ts         # Contact manifold → constraint rows
│   ├── joint.ts           # Revolute/fixed joint constraints
│   └── spring.ts          # Distance spring constraints
├── 2d/
│   ├── collision-sat.ts   # 2D SAT + circle collision
│   └── index.ts           # 2D public API (Rapier-style)
├── 3d/
│   ├── collision-gjk.ts   # 3D SAT + sphere collision
│   └── index.ts           # 3D public API
└── index.ts               # Main entry point
```

## AVBD Algorithm

The solver implements the Augmented Vertex Block Descent algorithm per timestep:

1. **Broadphase**: AABB overlap test between all body pairs
2. **Narrowphase**: SAT (boxes) / analytic (spheres) contact generation
3. **Warmstart**: Decay penalty parameters (γ), clamp to [penalty_min, penalty_max]
4. **Initialize bodies**: Compute inertial targets (x̂ = x + v·dt + g·dt²)
5. **Solver loop** (N iterations + optional post-stabilization):
   - **Primal update**: Per body, build M/dt² + Σ(JJᵀρ + G) system, solve via LDL^T
   - **Dual update**: Update λ = clamp(ρC + λ, fmin, fmax), ramp ρ += β|C|
6. **Velocity recovery**: v = (x - x_prev) / dt at iteration N-1

## References

- [VBD Paper (SIGGRAPH 2024)](https://arxiv.org/abs/2403.06321) — Foundational algorithm
- [AVBD Paper (SIGGRAPH 2025)](https://graphics.cs.utah.edu/research/projects/avbd/) — Augmented Lagrangian extension
- [avbd-demo2d](https://github.com/savant117/avbd-demo2d) — Primary reference implementation (MIT)
- [avbd-demo3d](https://github.com/savant117/avbd-demo3d) — 3D reference (MIT)
- [Rapier.js](https://rapier.rs/) — API design reference

## Development

```bash
npm install
npm test           # Run all 151 tests
npm run test:watch # Watch mode
npm run dev        # Start demo server (Vite)
npm run build      # Build TypeScript
```

## License

MIT
